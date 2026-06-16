use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NewickError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

pub struct TreeArrays {
    pub adjacency: Vec<Vec<f32>>, // Dense adjacency for TREX: n_nodes x n_nodes
    pub parents: Vec<i32>,
    pub post_order: Vec<i32>,
    pub leaf_names: Vec<String>,
    pub n_leaves: usize,
}

struct RawNode {
    name: String,
    parent: Option<usize>,
    children: Vec<usize>,
    #[allow(dead_code)]
    is_leaf: bool,
}

pub fn parse_newick<P: AsRef<Path>>(path: P) -> Result<TreeArrays, NewickError> {
    let content = fs::read_to_string(path)?;
    let content = content.trim();
    if !content.ends_with(';') {
        return Err(NewickError::InvalidFormat(
            "Newick must end with ';'".into(),
        ));
    }

    let content = &content[..content.len() - 1]; // strip ;

    let mut nodes: Vec<RawNode> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut current_name = String::new();

    // Simple state machine for Newick parsing
    let chars: Vec<char> = content.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '(' => {
                // Start a new internal node
                let node_idx = nodes.len();
                nodes.push(RawNode {
                    name: String::new(),
                    parent: stack.last().cloned(),
                    children: Vec::new(),
                    is_leaf: false,
                });
                if let Some(&p_idx) = stack.last() {
                    nodes[p_idx].children.push(node_idx);
                }
                stack.push(node_idx);
                i += 1;
            }
            ')' | ',' => {
                // If we have a name (leaf name or internal label), finalize it
                if !current_name.is_empty() {
                    let node_idx = nodes.len();
                    nodes.push(RawNode {
                        name: current_name.clone(),
                        parent: stack.last().cloned(),
                        children: Vec::new(),
                        is_leaf: true,
                    });
                    if let Some(&p_idx) = stack.last() {
                        nodes[p_idx].children.push(node_idx);
                    }
                    current_name.clear();
                }

                if chars[i] == ')' {
                    // Close the current internal node
                    stack.pop();
                    // Internal labels can follow ')'
                    i += 1;
                    while i < chars.len() && !matches!(chars[i], '(' | ')' | ',' | ':' | ';') {
                        nodes[*stack.last().unwrap_or(&0)].name.push(chars[i]);
                        i += 1;
                    }
                    continue; // Skip the i+=1 at bottom
                }
                i += 1;
            }
            ':' => {
                // Skip branch lengths for now
                i += 1;
                while i < chars.len() && !matches!(chars[i], '(' | ')' | ',' | ';') {
                    i += 1;
                }
            }
            ' ' | '\n' | '\t' | '\r' => {
                i += 1;
            }
            _ => {
                current_name.push(chars[i]);
                i += 1;
            }
        }
    }

    // Finalize if root was a single leaf (rare)
    if !current_name.is_empty() {
        nodes.push(RawNode {
            name: current_name,
            parent: None,
            children: Vec::new(),
            is_leaf: true,
        });
    }

    // Post-process into JAX-compatible arrays
    // Leaves first, then internal nodes
    let mut leaf_indices = Vec::new();
    let mut internal_indices = Vec::new();
    for (idx, node) in nodes.iter().enumerate() {
        if node.children.is_empty() {
            leaf_indices.push(idx);
        } else {
            internal_indices.push(idx);
        }
    }

    let n_leaves = leaf_indices.len();
    let n_internal = internal_indices.len();
    let n_all = n_leaves + n_internal;

    // Map raw node indices to output indices [leaves..., internal...]
    let mut raw_to_out = vec![0; nodes.len()];
    let mut leaf_names = Vec::new();
    for (i, &raw_idx) in leaf_indices.iter().enumerate() {
        raw_to_out[raw_idx] = i;
        leaf_names.push(nodes[raw_idx].name.clone());
    }
    for (i, &raw_idx) in internal_indices.iter().enumerate() {
        raw_to_out[raw_idx] = n_leaves + i;
    }

    let mut parents = vec![-1; n_all];
    let mut adjacency = vec![vec![0.0; n_all]; n_all];

    for (idx, node) in nodes.iter().enumerate() {
        let out_idx = raw_to_out[idx];
        if let Some(p_raw) = node.parent {
            let p_out = raw_to_out[p_raw];
            parents[out_idx] = p_out as i32;
            adjacency[out_idx][p_out] = 1.0;
        } else {
            // Root
            parents[out_idx] = out_idx as i32;
            adjacency[out_idx][out_idx] = 1.0;
        }
    }

    // Compute post-order
    let mut post_order = Vec::new();
    let _visited = vec![false; nodes.len()];
    let root_raw = nodes.iter().position(|r| r.parent.is_none()).unwrap_or(0);

    // Simple iterative post-order traversal
    let mut work_stack = vec![(root_raw, false)];
    while let Some((idx, processed)) = work_stack.pop() {
        if processed {
            post_order.push(raw_to_out[idx] as i32);
        } else {
            work_stack.push((idx, true));
            for &child in nodes[idx].children.iter().rev() {
                work_stack.push((child, false));
            }
        }
    }

    Ok(TreeArrays {
        adjacency,
        parents,
        post_order,
        leaf_names,
        n_leaves,
    })
}
