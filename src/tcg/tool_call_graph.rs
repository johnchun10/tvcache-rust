use crate::tcg::Node;
use crate::types::SandboxCandidate;
use crate::utils;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, Serialize)]
pub struct ToolCallGraph {
    pub root: Node,
    pub number_of_nodes: u32,
    pub number_of_sandboxes: u32,
    pub size_in_bytes: u64,
    // sandbox_id -> reference count; only positive counts are stored, an
    // absent entry means a count of zero.
    pub sandbox_references: BTreeMap<String, u32>,
    pub reclaimable_sandboxes: BTreeSet<String>, // Sandboxes marked for eviction (by the sandbox manager or the internal eviction loop) that are waiting to be handed to the deletion loop
    pub fork_cost_ema_ms: u32, // EMA of client-reported sandbox fork times for this task, in milliseconds, u32 is ok here because, it can store up to 4 million seconds which is > 40 days.
}

impl ToolCallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        path: Vec<String>,
        value: Option<String>,
        sandbox_id: Option<String>,
        runtime: u32,
        ttl: Option<u32>,
        ttl_exists: bool,
        embedding: Option<Vec<f32>>,
    ) -> (bool, bool) {
        if path.is_empty() {
            return (false, false);
        }

        let last_idx = path.len() - 1;
        let mut curr: &mut Node = &mut self.root;

        for (idx, p) in path.into_iter().enumerate() {
            if idx < last_idx {
                if let Some(next) = curr.child_mut(&p) {
                    curr = next;
                } else {
                    // Not inserting, so no need to call update size.
                    return (false, false);
                }
            } else {
                let mut sandbox_id_consumed = false;

                let (new_node, is_new_node) = curr.get_or_create_child(p.clone());
                let old_size_in_bytes = new_node.size_in_bytes;
                if is_new_node {
                    self.number_of_nodes += 1;
                }
                let entry = new_node.tool_calls.get_mut(&p).expect("Missing tool call");

                // Write rules (see plans/04-correctness-framework.md, bug B2):
                //  - Values are first-writer-wins. A different value for a node
                //    that already has one is rejected, and so is any sandbox
                //    attached with it: that snapshot did not produce the cached
                //    value. An identical value agrees and is not a conflict.
                //  - inserted_at_seconds (the TTL clock) only moves when a value
                //    is actually written, so a replay cannot renew a TTL.
                //  - Absent fields leave existing node state untouched, so an
                //    insert with only a sandbox cannot destroy a cached value.
                //  - Sandboxes are first-writer-wins.
                let value_conflict = !is_new_node
                    && matches!((&entry.value, &value), (Some(old), Some(new)) if old != new);

                if is_new_node || (value.is_some() && entry.value.is_none()) {
                    new_node.inserted_at_seconds = utils::current_utc_seconds();
                    entry.value = value;
                    entry.runtime = runtime;
                }

                if ttl_exists && !value_conflict {
                    new_node.ttl_present = true;

                    if let Some(ttl) = ttl {
                        new_node.ttl = ttl;
                    }
                }

                if sandbox_id.is_some() && new_node.sandbox_id.is_none() && !value_conflict {
                    new_node.sandbox_id = sandbox_id;
                    self.number_of_sandboxes += 1;
                    sandbox_id_consumed = true;
                }

                // A tool call's embedding never changes, so keep the first one.
                if embedding.is_some() && entry.embedding.is_none() {
                    entry.embedding = embedding;
                    new_node.settled = false;
                }

                new_node.update_size_in_bytes();
                self.size_in_bytes =
                    self.size_in_bytes - old_size_in_bytes as u64 + new_node.size_in_bytes as u64;

                return (!value_conflict, sandbox_id_consumed);
            }
        }

        (false, false)
    }

    pub fn find(
        &mut self,
        path: Vec<String>,
    ) -> (Option<String>, (Option<Vec<String>>, Option<String>)) {
        // this function tries to find the exact path in the tcg and return the value of the last node or return the longest prefix match with a sandbox id

        // Walk once, bumping the cache-hit counter on a full match. Nothing in
        // here can panic: a node reached through a call always has that call's
        // entry, and a missing one reads as a miss instead of an index panic.
        let path_len = path.len();
        let mut value_of_path: Option<String> = None;

        let (sandbox_path, sandbox_id) = self.walk_path(&path, |node, depth| {
            if depth == path_len {
                node.number_of_cache_hits += 1;
                value_of_path = node
                    .tool_calls
                    .get(&path[depth - 1])
                    .and_then(|entry| entry.value.clone());
            }
        });

        // Second walk: bump the LPM counter. node_at_mut returns None instead of
        // panicking, so a bookkeeping bug can never kill the connection while
        // the TCG lock is held (bug B4).
        if value_of_path.is_none() {
            if let Some(sandbox_path) = &sandbox_path {
                if let Some(node) = node_at_mut(&mut self.root, sandbox_path) {
                    node.number_of_lpm_hits += 1;
                }
            }
        }

        (value_of_path, (sandbox_path, sandbox_id))
    }

    /// Returns the deepest snapshot on path and references it, without counting hits.
    pub fn lease_sandbox(&mut self, path: &[String]) -> (Option<Vec<String>>, Option<String>) {
        let lease = self.walk_path(path, |_, _| {});
        if let Some(sandbox_id) = &lease.1 {
            self.increment_sandbox_reference(sandbox_id);
        }
        lease
    }

    /// Follows path while the graph matches it, visiting each matched node with its depth.
    /// Returns the deepest snapshot on the matched prefix.
    fn walk_path<F>(
        &mut self,
        path: &[String],
        mut visit: F,
    ) -> (Option<Vec<String>>, Option<String>)
    where
        F: FnMut(&mut Node, usize),
    {
        let mut curr = &mut self.root;
        let mut sandbox_path = None;
        let mut sandbox_id = None;

        // if let Some(id) = &curr.sandbox_id {
        //     sandbox_id = Some(id.clone());
        // } // No need to store root's sandbox id

        for (index, p) in path.iter().enumerate() {
            // The path has diverged from the tcg, so return what has been seen so far.
            let Some(next) = curr.child_mut(p) else {
                break;
            };
            curr = next;
            visit(curr, index + 1);

            if let Some(id) = &curr.sandbox_id {
                sandbox_id = Some(id.clone());
                sandbox_path = Some(path[..=index].to_vec());
            }
        }

        (sandbox_path, sandbox_id)
    }

    pub fn collect_sandbox_candidates(&self, task_id: &str) -> Vec<SandboxCandidate> {
        let mut candidates = Vec::new();
        let mut current_path = Vec::new();

        collect_sandbox_candidates_recursive(
            task_id,
            &self.root,
            &mut current_path,
            &mut candidates,
        );

        candidates
    }

    pub fn increment_sandbox_reference(&mut self, sandbox_id: &str) -> u32 {
        let reference_count = self
            .sandbox_references
            .entry(sandbox_id.to_string())
            .or_insert(0);
        *reference_count += 1;
        *reference_count
    }

    pub fn decrement_sandbox_reference(&mut self, sandbox_id: &str) -> u32 {
        let Some(reference_count) = self.sandbox_references.get_mut(sandbox_id) else {
            return 0;
        };

        *reference_count -= 1;

        if *reference_count == 0 {
            self.sandbox_references.remove(sandbox_id);
            return 0;
        }

        *reference_count
    }

    pub fn add_reclaimable_sandboxes(&mut self, sandbox_ids: &[String]) {
        for sandbox_id in sandbox_ids {
            self.reclaimable_sandboxes.insert(sandbox_id.clone());
        }
    }

    pub fn mark_sandbox_evicted(&mut self, sandbox_path: &[String], sandbox_id: &str) {
        // Always record the victim, even if the node is gone or mismatched,
        // so the id flows back to the deletion loop through drain instead of leaking.
        self.reclaimable_sandboxes.insert(sandbox_id.to_string());

        let mut curr: &mut Node = &mut self.root;

        for p in sandbox_path {
            if let Some(next) = curr.child_mut(p) {
                curr = next;
            } else {
                // Node already removed (this can happen by TTL eviction or TCG size based eviction that runs in the background).
                return;
            }
        }

        // as_deref turns Option<String> into Option<&str> for an allocation-free compare.
        if sandbox_path.is_empty() || curr.sandbox_id.as_deref() != Some(sandbox_id) {
            return;
        }

        curr.sandbox_id = None;
        self.number_of_sandboxes -= 1;

        let old_size_in_bytes = curr.size_in_bytes;

        curr.update_size_in_bytes();

        self.size_in_bytes =
            self.size_in_bytes - old_size_in_bytes as u64 + curr.size_in_bytes as u64;
    }

    pub fn drain_reclaimable_sandboxes(&mut self) -> Vec<String> {
        // Victims still referenced by a rollout stay in the set for a later drain.
        let sandbox_references = &self.sandbox_references;
        let mut ready = Vec::new();

        self.reclaimable_sandboxes.retain(|sandbox_id| {
            if sandbox_references.contains_key(sandbox_id) {
                true
            } else {
                ready.push(sandbox_id.clone());
                false
            }
        });

        ready
    }

    pub fn record_fork_cost(&mut self, sample_ms: u32) -> u32 {
        // Integer EMA with alpha = 0.3; the first sample seeds it. u64
        // intermediate so 7 * ema cannot overflow u32.
        // EMA means exponential moving average.
        if self.fork_cost_ema_ms == 0 {
            self.fork_cost_ema_ms = sample_ms;
        } else {
            self.fork_cost_ema_ms =
                ((7 * self.fork_cost_ema_ms as u64 + 3 * sample_ms as u64) / 10) as u32;
        }

        self.fork_cost_ema_ms
    }
}

fn node_at_mut<'a>(root: &'a mut Node, path: &[String]) -> Option<&'a mut Node> {
    let mut curr = root;

    for p in path {
        curr = curr.child_mut(p)?;
    }

    Some(curr)
}

fn collect_sandbox_candidates_recursive(
    task_id: &str,
    node: &Node,
    current_path: &mut Vec<String>,
    candidates: &mut Vec<SandboxCandidate>,
) {
    if let Some(sandbox_id) = &node.sandbox_id {
        candidates.push(SandboxCandidate {
            task_id: task_id.to_string(),
            sandbox_path: current_path.clone(),
            sandbox_id: sandbox_id.clone(),
            score: node.number_of_lpm_hits,
        });
    }

    for (path_entry, child) in node.iter_children() {
        current_path.push(path_entry.clone());
        collect_sandbox_candidates_recursive(task_id, child, current_path, candidates);
        current_path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn per_call_entries_preserve_insert_semantics() {
        let mut graph = ToolCallGraph::new();
        assert!(graph.root.tool_calls.is_empty());
        graph.insert(
            vec!["a".into()],
            Some("value".into()),
            Some("snapshot".into()),
            7,
            Some(60),
            true,
            None,
        );
        let original_size = graph.size_in_bytes;
        for (embedding, settled) in [
            (None, true),
            (Some(vec![1.0, 0.0]), false),
            (Some(vec![0.0, 1.0]), true),
            (None, true),
        ] {
            graph.root.child_mut("a").unwrap().settled = true;
            assert_eq!(
                graph.insert(
                    vec!["a".into()],
                    None,
                    Some("replacement".into()),
                    99,
                    None,
                    false,
                    embedding,
                ),
                (true, false)
            );
            assert_eq!(graph.root.child("a").unwrap().settled, settled);
        }
        assert_eq!(graph.find(vec!["a".into()]).0.as_deref(), Some("value"));
        assert_eq!((graph.number_of_nodes, graph.number_of_sandboxes), (1, 1));
        assert_eq!(graph.size_in_bytes, original_size + 8);
        let node = graph.root.child_mut("a").unwrap();
        assert_eq!(node.sandbox_id.as_deref(), Some("snapshot"));
        assert!(node.ttl_present && node.ttl == 60);
        assert_eq!(node.tool_calls["a"].runtime, 7);
        assert_eq!(node.tool_calls["a"].embedding, Some(vec![1.0, 0.0]));
        let original_size = node.size_in_bytes;
        node.tool_calls
            .insert("alias".into(), node.tool_calls["a"].clone());
        node.update_size_in_bytes();
        assert_eq!(node.size_in_bytes, original_size + 5 + 5 + 4 + 8);
    }

    #[test]
    fn lease_returns_deepest_snapshot_without_counting_hits() {
        let path = |calls: &[&str]| {
            calls
                .iter()
                .map(|call| call.to_string())
                .collect::<Vec<_>>()
        };
        let mut graph = ToolCallGraph::new();
        graph.insert(
            path(&["a"]),
            Some("a".into()),
            Some("shallow".into()),
            1,
            None,
            false,
            None,
        );
        graph.insert(
            path(&["a", "b"]),
            Some("b".into()),
            Some("deep".into()),
            1,
            None,
            false,
            None,
        );
        graph.insert(
            path(&["a", "b", "c"]),
            Some("c".into()),
            None,
            1,
            None,
            false,
            None,
        );

        let deep = (Some(path(&["a", "b"])), Some("deep".to_string()));
        assert_eq!(graph.lease_sandbox(&path(&["a", "b", "c"])), deep);
        assert_eq!(graph.lease_sandbox(&path(&["a", "b", "missing"])), deep);
        assert_eq!(graph.lease_sandbox(&path(&["missing"])), (None, None));
        assert_eq!(graph.lease_sandbox(&[]), (None, None));
        assert_eq!(graph.sandbox_references["deep"], 2);
        assert!(!graph.sandbox_references.contains_key("shallow"));
        let hits = |graph: &ToolCallGraph| {
            graph.root.walk_subtcg(0, &mut |acc, node| {
                acc + node.number_of_cache_hits + node.number_of_lpm_hits
            })
        };
        assert_eq!(hits(&graph), 0);

        assert_eq!(
            graph.find(path(&["a", "b", "c"])),
            (Some("c".into()), deep.clone())
        );
        assert_eq!(
            graph.find(path(&["a", "b"])),
            (Some("b".into()), deep.clone())
        );
        assert_eq!(graph.find(path(&["a", "b", "missing"])), (None, deep));
        assert_eq!(graph.find(Vec::new()), (None, (None, None)));
        let b = graph.root.child("a").unwrap().child("b").unwrap();
        assert_eq!((b.number_of_cache_hits, b.number_of_lpm_hits), (1, 1));
        assert_eq!(b.child("c").unwrap().number_of_cache_hits, 1);
        assert_eq!(hits(&graph), 3);
        assert_eq!(graph.sandbox_references["deep"], 2);
    }

    fn p(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|e| e.to_string()).collect()
    }

    fn insert(
        tcg: &mut ToolCallGraph,
        path: &[&str],
        value: Option<&str>,
        sandbox: Option<&str>,
    ) -> (bool, bool) {
        tcg.insert(
            p(path),
            value.map(str::to_string),
            sandbox.map(str::to_string),
            1,
            None,
            false,
            None,
        )
    }

    fn assert_consistent(tcg: &ToolCallGraph) {
        let violations = tcg.check_invariants();
        assert!(violations.is_empty(), "invariants violated: {violations:?}");
    }

    #[test]
    fn insert_requires_parent_and_rejects_empty_path() {
        let mut tcg = ToolCallGraph::new();
        assert_eq!(insert(&mut tcg, &[], Some("v"), None), (false, false));
        assert_eq!(insert(&mut tcg, &["a", "b"], Some("v"), Some("s")), (false, false));
        assert_eq!(tcg.number_of_nodes, 0);
        assert_eq!(tcg.number_of_sandboxes, 0);
        assert_consistent(&tcg);
    }

    #[test]
    fn value_is_first_writer_wins() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("first"), None);
        let (accepted, _) = insert(&mut tcg, &["a"], Some("second"), None);

        assert!(!accepted, "a conflicting value must be rejected");
        let (value, _) = tcg.find(p(&["a"]));
        assert_eq!(value.as_deref(), Some("first"));
        assert_consistent(&tcg);
    }

    #[test]
    fn identical_value_is_accepted_and_does_not_renew_the_ttl_clock() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("v"), None);
        tcg.root.child_mut("a").unwrap().inserted_at_seconds = 5;

        let (accepted, _) = insert(&mut tcg, &["a"], Some("v"), None);

        assert!(accepted);
        assert_eq!(tcg.root.child("a").unwrap().inserted_at_seconds, 5);
    }

    #[test]
    fn conflicting_value_cannot_attach_a_sandbox() {
        // Bug B2: a snapshot from a run that produced a different value must
        // never sit next to the first run's value.
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("run-a"), None);

        let result = insert(&mut tcg, &["a"], Some("run-b"), Some("snap-b"));

        assert_eq!(result, (false, false));
        assert_eq!(tcg.number_of_sandboxes, 0);
        assert!(tcg.root.child("a").unwrap().sandbox_id.is_none());
        assert_consistent(&tcg);
    }

    #[test]
    fn agreeing_value_attaches_sandbox_once() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("v"), None);

        assert_eq!(insert(&mut tcg, &["a"], Some("v"), Some("s1")), (true, true));
        assert_eq!(insert(&mut tcg, &["a"], Some("v"), Some("s2")), (true, false));
        assert_eq!(tcg.root.child("a").unwrap().sandbox_id.as_deref(), Some("s1"));
        assert_eq!(tcg.number_of_sandboxes, 1);
        assert_consistent(&tcg);
    }

    #[test]
    fn sandbox_only_insert_keeps_value_and_value_fills_a_valueless_node() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("v"), None);
        insert(&mut tcg, &["a"], None, Some("s"));
        assert_eq!(
            tcg.root.child("a").unwrap().tool_calls["a"].value.as_deref(),
            Some("v")
        );

        // A placeholder (no value) is later filled in by a real insert.
        insert(&mut tcg, &["b"], None, None);
        assert!(tcg.find(p(&["b"])).0.is_none());
        insert(&mut tcg, &["b"], Some("late"), None);
        assert_eq!(tcg.find(p(&["b"])).0.as_deref(), Some("late"));
        assert_consistent(&tcg);
    }

    #[test]
    fn find_returns_deepest_sandbox_on_the_walked_prefix() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("1"), Some("s-a"));
        insert(&mut tcg, &["a", "b"], Some("2"), None);
        insert(&mut tcg, &["a", "b", "c"], Some("3"), Some("s-c"));

        let (value, (path, id)) = tcg.find(p(&["a", "b", "c", "d"]));
        assert_eq!(value, None);
        assert_eq!(path, Some(p(&["a", "b", "c"])));
        assert_eq!(id.as_deref(), Some("s-c"));

        let (value, (path, id)) = tcg.find(p(&["a", "b", "x"]));
        assert_eq!(value, None);
        assert_eq!(path, Some(p(&["a"])));
        assert_eq!(id.as_deref(), Some("s-a"));

        let (value, (path, id)) = tcg.find(p(&["zzz"]));
        assert_eq!((value, path, id), (None, None, None));
    }

    #[test]
    fn find_counts_cache_hits_and_lpm_hits_separately() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("1"), Some("s"));
        insert(&mut tcg, &["a", "b"], Some("2"), None);

        tcg.find(p(&["a", "b"])); // full hit
        tcg.find(p(&["a", "b", "c"])); // lpm hit at "a"
        tcg.find(p(&["a", "q"])); // lpm hit at "a"

        let a = tcg.root.child("a").unwrap();
        assert_eq!(a.child("b").unwrap().number_of_cache_hits, 1);
        assert_eq!(a.number_of_lpm_hits, 2);
        assert_eq!(a.number_of_cache_hits, 0);
    }

    #[test]
    fn sandbox_references_and_drain() {
        let mut tcg = ToolCallGraph::new();
        assert_eq!(tcg.increment_sandbox_reference("s"), 1);
        assert_eq!(tcg.increment_sandbox_reference("s"), 2);
        tcg.add_reclaimable_sandboxes(&["s".to_string(), "free".to_string()]);

        assert_eq!(tcg.drain_reclaimable_sandboxes(), vec!["free".to_string()]);
        assert_eq!(tcg.decrement_sandbox_reference("s"), 1);
        assert!(tcg.drain_reclaimable_sandboxes().is_empty());
        assert_eq!(tcg.decrement_sandbox_reference("s"), 0);
        assert_eq!(tcg.drain_reclaimable_sandboxes(), vec!["s".to_string()]);
        assert_eq!(tcg.decrement_sandbox_reference("never-seen"), 0);
        assert_consistent(&tcg);
    }

    #[test]
    fn mark_evicted_handles_missing_and_mismatched_nodes() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("1"), Some("s"));

        // Mismatched id: node keeps its sandbox, but the victim is still recorded.
        tcg.mark_sandbox_evicted(&p(&["a"]), "other");
        assert_eq!(tcg.number_of_sandboxes, 1);
        assert!(tcg.reclaimable_sandboxes.contains("other"));

        // Missing node: no panic, victim recorded.
        tcg.mark_sandbox_evicted(&p(&["gone"]), "ghost");
        assert!(tcg.reclaimable_sandboxes.contains("ghost"));

        tcg.mark_sandbox_evicted(&p(&["a"]), "s");
        assert_eq!(tcg.number_of_sandboxes, 0);
        assert!(tcg.root.child("a").unwrap().sandbox_id.is_none());
        assert_consistent(&tcg);
    }

    #[test]
    fn fork_cost_ema() {
        let mut tcg = ToolCallGraph::new();
        assert_eq!(tcg.record_fork_cost(100), 100);
        assert_eq!(tcg.record_fork_cost(200), (7 * 100 + 3 * 200) / 10);
    }

    #[test]
    fn invariant_checker_detects_drift() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["a"], Some("1"), Some("s"));
        assert_consistent(&tcg);

        tcg.number_of_nodes += 1;
        tcg.size_in_bytes += 1;
        tcg.reclaimable_sandboxes.insert("s".to_string());
        let violations = tcg.check_invariants();
        assert_eq!(violations.len(), 3, "{violations:?}");
    }

    /// Deterministic xorshift so failures reproduce from the seed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Model-based test: random inserts and finds against a trivially correct
    /// map-based model of the documented semantics.
    #[test]
    fn random_operations_match_a_reference_model() {
        #[derive(Default, Clone)]
        struct ModelNode {
            value: Option<String>,
            sandbox: Option<String>,
        }

        for seed in 1..=200u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            let mut tcg = ToolCallGraph::new();
            let mut model: HashMap<Vec<String>, ModelNode> = HashMap::new();
            let keys = ["a", "b", "c"];

            for step in 0..300 {
                let depth = 1 + rng.below(4) as usize;
                let path: Vec<String> = (0..depth)
                    .map(|_| keys[rng.below(keys.len() as u64) as usize].to_string())
                    .collect();

                if rng.below(3) < 2 {
                    let value = if rng.below(5) == 0 {
                        None
                    } else {
                        Some(format!("v{}", rng.below(3)))
                    };
                    let sandbox = if rng.below(2) == 0 {
                        Some(format!("s{seed}-{step}"))
                    } else {
                        None
                    };

                    let parent_ok = path.len() == 1 || model.contains_key(&path[..path.len() - 1]);
                    let (got_ok, got_sandbox) =
                        tcg.insert(path.clone(), value.clone(), sandbox.clone(), 1, None, false, None);

                    if !parent_ok {
                        assert_eq!((got_ok, got_sandbox), (false, false), "seed {seed}");
                        continue;
                    }

                    let node = model.entry(path.clone()).or_default();
                    let conflict = matches!((&node.value, &value), (Some(a), Some(b)) if a != b);
                    if !conflict && node.value.is_none() {
                        node.value = value.clone();
                    }
                    let mut want_sandbox = false;
                    if !conflict && sandbox.is_some() && node.sandbox.is_none() {
                        node.sandbox = sandbox.clone();
                        want_sandbox = true;
                    }
                    assert_eq!((got_ok, got_sandbox), (!conflict, want_sandbox), "seed {seed} step {step}");
                } else {
                    let (value, (sandbox_path, sandbox_id)) = tcg.find(path.clone());

                    let want_value = model.get(&path).and_then(|n| n.value.clone());
                    let mut want_sandbox: Option<(Vec<String>, String)> = None;
                    for len in 1..=path.len() {
                        match model.get(&path[..len]) {
                            Some(node) => {
                                if let Some(s) = &node.sandbox {
                                    want_sandbox = Some((path[..len].to_vec(), s.clone()));
                                }
                            }
                            None => break,
                        }
                    }

                    assert_eq!(value, want_value, "seed {seed} step {step}");
                    assert_eq!(
                        sandbox_path.zip(sandbox_id),
                        want_sandbox,
                        "seed {seed} step {step}"
                    );
                }

                assert_consistent(&tcg);
            }
        }
    }
}
