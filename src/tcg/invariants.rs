//! Self-check of a ToolCallGraph's bookkeeping against a fresh recount.
//!
//! The counters (`number_of_nodes`, `number_of_sandboxes`, `size_in_bytes`) are
//! maintained incrementally by insert, eviction and sandbox marking; this
//! recomputes them from the tree so tests (and, later, a CheckInvariants
//! request) can detect drift.
//!
//! Only bookkeeping and structure are checked. Which run produced a node's
//! value and which produced its snapshot is not recorded, so it is not checked:
//! insert keeps the two from the same run, while a contraction merge may pair
//! one run's value with another run's snapshot, which is intended because the
//! merged states compared equal (see `merge` in contraction.rs).

use std::collections::BTreeSet;

use super::{Node, ToolCallGraph};

impl ToolCallGraph {
    /// Returns a human-readable description of every violated invariant;
    /// empty means the graph is consistent.
    #[allow(dead_code)]
    pub fn check_invariants(&self) -> Vec<String> {
        let mut violations = Vec::new();
        let mut totals = Totals::default();

        walk(&self.root, "", true, &mut totals, &mut violations);

        if totals.nodes != self.number_of_nodes {
            violations.push(format!(
                "number_of_nodes is {} but the tree has {}",
                self.number_of_nodes, totals.nodes
            ));
        }
        if totals.sandboxes != self.number_of_sandboxes {
            violations.push(format!(
                "number_of_sandboxes is {} but the tree has {}",
                self.number_of_sandboxes, totals.sandboxes
            ));
        }
        if totals.size_in_bytes != self.size_in_bytes {
            violations.push(format!(
                "size_in_bytes is {} but the tree sums to {}",
                self.size_in_bytes, totals.size_in_bytes
            ));
        }

        for sandbox_id in &totals.duplicate_sandbox_ids {
            violations.push(format!("sandbox id {sandbox_id} is attached to more than one node"));
        }

        for sandbox_id in &self.reclaimable_sandboxes {
            if totals.sandbox_ids.contains(sandbox_id) {
                violations.push(format!(
                    "sandbox id {sandbox_id} is reclaimable but still attached to a node"
                ));
            }
        }

        for (sandbox_id, count) in &self.sandbox_references {
            if *count == 0 {
                violations.push(format!("sandbox_references stores a zero count for {sandbox_id}"));
            }
        }

        violations
    }
}

#[derive(Default)]
struct Totals {
    nodes: u32,
    sandboxes: u32,
    size_in_bytes: u64,
    sandbox_ids: BTreeSet<String>,
    duplicate_sandbox_ids: BTreeSet<String>,
}

fn walk(node: &Node, key: &str, is_root: bool, totals: &mut Totals, violations: &mut Vec<String>) {
    if !is_root {
        totals.nodes += 1;
        totals.size_in_bytes += node.size_in_bytes as u64;

        let expected = node.computed_size_in_bytes();
        if expected != node.size_in_bytes {
            violations.push(format!(
                "node {key:?} has size_in_bytes {} but recomputes to {expected}",
                node.size_in_bytes
            ));
        }

        if let Some(sandbox_id) = &node.sandbox_id {
            totals.sandboxes += 1;
            if !totals.sandbox_ids.insert(sandbox_id.clone()) {
                totals.duplicate_sandbox_ids.insert(sandbox_id.clone());
            }
        }
    }

    // Contraction lets one node answer to several tool calls, so siblings
    // are checked for sharing a call: lookups would be ambiguous otherwise.
    let mut calls_seen = BTreeSet::new();
    for (child_key, child) in node.iter_children() {
        for call in child.tool_calls.keys() {
            if !calls_seen.insert(call) {
                violations.push(format!(
                    "tool call {call:?} names more than one child of node {key:?}"
                ));
            }
        }
        walk(child, child_key, false, totals, violations);
    }
}
