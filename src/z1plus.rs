//! Insertion-based implementation of the Z1+ SMDP core.
//!
//! Unlike the legacy remove/slide minimizer, this implementation keeps stable
//! node identities and can replace one middle node by multiple contact/ghost
//! nodes.  That state transition is required by the published Z1+ algorithm
//! and by official cases whose final kink count exceeds `N - 2`.

use molar::prelude::*;
use nalgebra::Vector3;
use std::collections::{HashSet, VecDeque};

use crate::z1::{ChainResult, FrameResult, MIN_TRUE_CHAIN};

type Vector3d = Vector3<f64>;
type NodeId = u64;

const GEOM_EPS: f64 = 1.0e-11;
const LENGTH_EPS: f64 = 1.0e-10;
const KINK_COS_DEVIATION: f64 = 0.001;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeKind {
    Original,
    Ghost,
    Contact,
    /// A former contact after Z1+'s second-pass B-dagger update.  It remains
    /// in the working path but is not necessarily a final kink.
    BarrierGhost,
    /// A fixed-obstacle contact rejected by the best-match pass. It must not
    /// be reintroduced by the later folded-path recovery pass.
    RejectedGhost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SegmentId {
    chain: usize,
    start: NodeId,
    end: NodeId,
}

#[derive(Clone, Copy, Debug)]
struct Contact {
    obstacle: SegmentId,
    obstacle_parameter: f64,
    /// Stable material coordinate (continuous contour position) of the contact
    /// along the obstacle chain. The node IDs in `obstacle` are transient — the
    /// obstacle chain's own minimization removes/renumbers them — but `s` is a
    /// material label that survives, so a mutual contact can be re-resolved to
    /// the obstacle's current geometry instead of being silently dropped.
    obstacle_s: f64,
}

#[derive(Clone, Debug)]
struct Node {
    id: NodeId,
    position: Vector3d,
    /// Continuous one-based position along the original chain.
    s: f64,
    kind: NodeKind,
    contact: Option<Contact>,
}

#[derive(Clone, Debug)]
struct WorkChain {
    original_n: usize,
    ree: f64,
    movable: bool,
    nodes: Vec<Node>,
}

#[derive(Clone, Debug)]
struct Network {
    chains: Vec<WorkChain>,
    next_id: NodeId,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub self_entanglement: bool,
    pub max_sweeps: usize,
    pub thickness: f64,
    /// Z1+'s maximum working-segment length. Infinite selects the maximum
    /// initial bond length, corresponding to `lmax_factor = 1`.
    pub lmax: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            self_entanglement: false,
            max_sweeps: 2000,
            thickness: 0.002,
            lmax: f64::INFINITY,
        }
    }
}

#[derive(Clone, Debug)]
struct Crossing {
    point: Vector3d,
    metric: f64,
    obstacle: SegmentId,
    obstacle_parameter: f64,
}

impl Network {
    fn from_input(chains: &[Vec<Vector3f>], pbox: Option<&PeriodicBox>) -> Self {
        let mut next_id = 1;
        let mut work = Vec::with_capacity(chains.len());
        for chain in chains {
            let nodes = chain
                .iter()
                .enumerate()
                .map(|(index, position)| {
                    let node = Node {
                        id: next_id,
                        position: position.cast::<f64>(),
                        s: index as f64 + 1.0,
                        kind: NodeKind::Original,
                        contact: None,
                    };
                    next_id += 1;
                    node
                })
                .collect::<Vec<_>>();
            let ree = if nodes.len() >= 2 {
                (nodes[nodes.len() - 1].position - nodes[0].position).norm()
            } else {
                0.0
            };
            work.push(WorkChain {
                original_n: chain.len(),
                ree,
                movable: chain.len() >= MIN_TRUE_CHAIN,
                nodes,
            });
        }
        let mut network = Self {
            chains: work,
            next_id,
        };
        network.introduce_nodes_to_reduce_cost(pbox);
        network
    }

    fn allocate_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Introduce nodes to reduce path cost: repeatedly
    /// bisect true-chain bonds longer than min(<bond>, density^(-1/3)).
    fn introduce_nodes_to_reduce_cost(&mut self, pbox: Option<&PeriodicBox>) {
        let Some(pbox) = pbox else {
            return;
        };
        let mut bond_sum = 0.0;
        let mut bond_count = 0usize;
        let atom_count: usize = self.chains.iter().map(|chain| chain.original_n).sum();
        for chain in self.chains.iter().filter(|chain| chain.movable) {
            for segment in chain.nodes.windows(2) {
                bond_sum += (segment[1].position - segment[0].position).norm();
                bond_count += 1;
            }
        }
        if bond_count == 0 {
            return;
        }
        let volume = pbox.get_matrix().cast::<f64>().determinant().abs();
        if volume <= 0.0 || !volume.is_finite() {
            return;
        }
        let density = atom_count as f64 / volume;
        let upper = (bond_sum / bond_count as f64).min(density.powf(-1.0 / 3.0));
        if upper <= 0.0 || !upper.is_finite() {
            return;
        }

        for chain_index in 0..self.chains.len() {
            if !self.chains[chain_index].movable {
                continue;
            }
            let old = std::mem::take(&mut self.chains[chain_index].nodes);
            let mut refined = Vec::with_capacity(old.len());
            refined.push(old[0].clone());
            for pair in old.windows(2) {
                let length = (pair[1].position - pair[0].position).norm();
                let mut pieces = 1usize;
                while length / pieces as f64 > upper {
                    pieces *= 2;
                }
                for piece in 1..pieces {
                    let t = piece as f64 / pieces as f64;
                    refined.push(Node {
                        id: self.allocate_id(),
                        position: pair[0].position + (pair[1].position - pair[0].position) * t,
                        s: pair[0].s + (pair[1].s - pair[0].s) * t,
                        kind: NodeKind::Ghost,
                        contact: None,
                    });
                }
                refined.push(pair[1].clone());
            }
            self.chains[chain_index].nodes = refined;
        }
    }
}

/// Analyze one frame using the insertion-based Z1+ core.
pub fn analyze_frame(
    chains: &[Vec<Vector3f>],
    pbox: Option<&PeriodicBox>,
    mut options: Options,
) -> FrameResult {
    if options.lmax.is_infinite() {
        let initial_max_bond = chains
            .iter()
            .flat_map(|chain| chain.windows(2))
            .map(|pair| (pair[1] - pair[0]).norm() as f64)
            .fold(0.0, f64::max);
        options.lmax = initial_max_bond;
        if let Some(pbox) = pbox {
            let extents = pbox.get_box_extents();
            let min_box = f64::from(extents.x.min(extents.y).min(extents.z));
            if initial_max_bond < 0.25 * min_box {
                let system_n = chains.iter().map(Vec::len).sum::<usize>() as f64;
                let density_lmax = min_box * (1.5 / system_n).powf(1.0 / 3.0);
                options.lmax = initial_max_bond.max(density_lmax);
            }
        }
    }
    let mut network = Network::from_input(chains, pbox);
    minimize(&mut network, pbox, options);
    if std::env::var_os("ENTANGL_Z1PLUS_TRACE").is_some() {
        for (chain_index, chain) in network.chains.iter().enumerate() {
            if chain.movable {
                let contacts = chain
                    .nodes
                    .iter()
                    .filter(|node| matches!(node.kind, NodeKind::Contact | NodeKind::BarrierGhost))
                    .count();
                eprintln!(
                    "TRACE prefinal chain={} nodes={} contacts={}",
                    chain_index + 1,
                    chain.nodes.len(),
                    contacts
                );
            }
        }
    }
    finalize_ghosts(&mut network, pbox, options);

    if std::env::var_os("ENTANGL_Z1PLUS_TRACE").is_some() {
        for (chain_index, chain) in network.chains.iter().enumerate() {
            if !chain.movable {
                continue;
            }
            eprintln!(
                "TRACE chain={} nodes={}",
                chain_index + 1,
                chain.nodes.len()
            );
            for (node_index, node) in chain.nodes.iter().enumerate() {
                let kink = node_index > 0
                    && node_index + 1 < chain.nodes.len()
                    && is_kink(
                        &chain.nodes[node_index - 1],
                        node,
                        &chain.nodes[node_index + 1],
                    );
                eprintln!(
                    "TRACE node={} id={} s={:.5} kind={:?} kink={} pos=({:.8},{:.8},{:.8}) contact={:?}",
                    node_index + 1,
                    node.id,
                    node.s,
                    node.kind,
                    kink,
                    node.position.x,
                    node.position.y,
                    node.position.z,
                    node.contact
                );
            }
        }
    }

    let chains = network
        .chains
        .iter()
        .map(|chain| {
            let lpp = chain
                .nodes
                .windows(2)
                .map(|pair| (pair[1].position - pair[0].position).norm())
                .sum::<f64>();
            let z = chain
                .nodes
                .windows(3)
                .filter(|triple| is_kink(&triple[0], &triple[1], &triple[2]))
                .count();
            ChainResult {
                n_beads: chain.original_n,
                is_true: chain.movable,
                z: if chain.movable { z } else { 0 },
                lpp: lpp as Float,
                ree: chain.ree as Float,
            }
        })
        .collect();
    FrameResult { chains }
}

fn minimize(network: &mut Network, pbox: Option<&PeriodicBox>, options: Options) {
    // Global node-pool sweep.
    //
    // A single per-chain relaxation tightens an isolated chain exactly, but lets
    // that chain race far ahead of its still-coiled neighbours and slip through
    // a moving obstacle — the dense mutual-entanglement leak. Z1+ instead
    // advances every node of the whole system together, so no chain gets ahead
    // of the others' current geometry.
    //
    // We queue every live interior node (by stable id, in contour order) and
    // process one at a time. After a change, the affected local neighbour is
    // requeued: for a single movable chain, at the FRONT (immediate
    // backtracking — reproduces the exact per-chain relaxation, so benchmarks
    // 01-04 stay exact); with several movable chains, at the BACK (its local
    // re-tightening is deferred behind the rest of the system, so chains
    // co-evolve and neither races into the other).
    let interleave = network.chains.iter().filter(|chain| chain.movable).count() > 1;
    for _ in 0..options.max_sweeps {
        let mut changed = false;
        // Snapshot the working list by stable id: removing a node cannot make
        // its successor run twice, and newly allocated Nprime contacts wait for
        // the next scan.
        let mut worklist = network
            .chains
            .iter()
            .enumerate()
            .flat_map(|(chain_index, chain)| {
                chain
                    .nodes
                    .iter()
                    .skip(1)
                    .take(chain.nodes.len().saturating_sub(2))
                    .map(move |node| (node.id, chain_index))
            })
            .collect::<VecDeque<_>>();
        let settle_contacts = network
            .chains
            .iter()
            .flat_map(|chain| &chain.nodes)
            .filter(|node| node.kind == NodeKind::Contact)
            .map(|node| node.id)
            .collect::<HashSet<_>>();

        while let Some((node_id, chain_index)) = worklist.pop_front() {
            if !network.chains[chain_index].movable {
                continue;
            }
            let Some(middle) = network.chains[chain_index]
                .nodes
                .iter()
                .position(|node| node.id == node_id)
            else {
                continue;
            };
            if middle == 0 || middle + 1 >= network.chains[chain_index].nodes.len() {
                continue;
            }
            let a = network.chains[chain_index].nodes[middle - 1].clone();
            let b = network.chains[chain_index].nodes[middle].clone();
            let c = network.chains[chain_index].nodes[middle + 1].clone();

            let mut revisiting_constraint = false;
            if b.kind == NodeKind::BarrierGhost {
                let moving_obstacle = b.contact.is_some_and(|contact| {
                    network
                        .chains
                        .get(contact.obstacle.chain)
                        .is_some_and(|chain| chain.movable)
                });
                if !moving_obstacle {
                    continue;
                }
                revisiting_constraint = true;
            }
            if b.kind == NodeKind::Contact {
                if !settle_contacts.contains(&b.id) {
                    continue;
                }
                if let Some(position) =
                    settled_contact_position(network, &a, &b, &c, pbox, options.thickness)
                {
                    let node = &mut network.chains[chain_index].nodes[middle];
                    if (node.position - position).norm() > LENGTH_EPS {
                        node.position = position;
                        changed = true;
                    }
                    node.kind = NodeKind::BarrierGhost;
                    continue;
                }
                let moving_obstacle = b.contact.is_some_and(|contact| {
                    network
                        .chains
                        .get(contact.obstacle.chain)
                        .is_some_and(|chain| chain.movable)
                });
                if !moving_obstacle {
                    continue;
                }
                revisiting_constraint = true;
            }

            let crossing = best_crossing(
                network,
                chain_index,
                middle,
                &a.position,
                &b.position,
                &c.position,
                pbox,
                options.self_entanglement,
            );
            match crossing {
                None => {
                    if !revisiting_constraint && (c.position - a.position).norm() > options.lmax {
                        let midpoint = (a.position + c.position) * 0.5;
                        let node = &mut network.chains[chain_index].nodes[middle];
                        if (node.position - midpoint).norm() > LENGTH_EPS {
                            node.position = midpoint;
                            node.kind = NodeKind::Ghost;
                            node.contact = None;
                            changed = true;
                        }
                    } else {
                        network.chains[chain_index].nodes.remove(middle);
                        changed = true;
                        // Re-examine the predecessor: the removal may have made
                        // it collapsible or exposed a new crossing.
                        if middle > 1 {
                            requeue(&mut worklist, a.id, chain_index, interleave);
                        }
                    }
                }
                Some(crossing) => {
                    let replacements =
                        local_replacement(network, &a, &b, &c, crossing, options.thickness);
                    if replacements.is_empty() || !replacement_shortens(&a, &b, &c, &replacements) {
                        continue;
                    }
                    network.chains[chain_index]
                        .nodes
                        .splice(middle..=middle, replacements);
                    changed = true;
                    // Re-examine the predecessor and the retained B-dagger node
                    // (id preserved by local_replacement); the freshly allocated
                    // contact waits for the next scan's rebuilt worklist. Order
                    // so the predecessor is processed before the B-dagger, as
                    // the per-chain relaxation would.
                    if interleave {
                        if middle > 1 {
                            worklist.push_back((a.id, chain_index));
                        }
                        worklist.push_back((b.id, chain_index));
                    } else {
                        worklist.push_front((b.id, chain_index));
                        if middle > 1 {
                            worklist.push_front((a.id, chain_index));
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Requeue a node's local re-tightening work: immediately (front) when a single
/// chain relaxes in isolation, deferred (back) when chains must interleave so
/// none races ahead of the others' current geometry.
fn requeue(worklist: &mut VecDeque<(NodeId, usize)>, node_id: NodeId, chain_index: usize, interleave: bool) {
    if interleave {
        worklist.push_back((node_id, chain_index));
    } else {
        worklist.push_front((node_id, chain_index));
    }
}

/// Material contour coordinate of a contact along its obstacle segment, from
/// the obstacle endpoints' `s` values. Stable across obstacle renumbering.
fn obstacle_s_from_segment(network: &Network, obstacle: SegmentId, parameter: f64) -> Option<f64> {
    let chain = network.chains.get(obstacle.chain)?;
    let s_start = chain.nodes.iter().find(|node| node.id == obstacle.start)?.s;
    let s_end = chain.nodes.iter().find(|node| node.id == obstacle.end)?.s;
    Some(s_start + (s_end - s_start) * parameter.clamp(0.0, 1.0))
}

/// Resolve a contact's current obstacle segment against the live network.
///
/// Fast path: the stored endpoint IDs still bound a live segment. Stable path:
/// the obstacle mutated (a node was removed, or a new node was inserted between
/// the stored endpoints), so re-locate the segment that currently spans the
/// stored material coordinate `obstacle_s`. Returns the imaged endpoints and
/// the in-segment parameter of the contact.
fn resolve_obstacle_segment(
    network: &Network,
    contact: &Contact,
    reference: &Vector3d,
    pbox: Option<&PeriodicBox>,
) -> Option<(Vector3d, Vector3d, f64)> {
    let chain = network.chains.get(contact.obstacle.chain)?;
    let adjacent_by_id = chain
        .nodes
        .iter()
        .position(|node| node.id == contact.obstacle.start)
        .filter(|&i| i + 1 < chain.nodes.len() && chain.nodes[i + 1].id == contact.obstacle.end);
    let (i0, i1, parameter) = if let Some(i0) = adjacent_by_id {
        (i0, i0 + 1, contact.obstacle_parameter)
    } else {
        // Bracket the stored material coordinate on the current contour.
        let s = contact.obstacle_s;
        let mut located = None;
        for w in 0..chain.nodes.len().saturating_sub(1) {
            let (sa, sb) = (chain.nodes[w].s, chain.nodes[w + 1].s);
            if (sb - sa).abs() > 0.0 && (sa - s) * (sb - s) <= 0.0 {
                located = Some((w, (s - sa) / (sb - sa)));
                break;
            }
        }
        let (w, p) = located?;
        (w, w + 1, p)
    };
    let (p0, p1) = image_segment(
        &chain.nodes[i0].position,
        &chain.nodes[i1].position,
        reference,
        pbox,
    );
    Some((p0, p1, parameter.clamp(0.0, 1.0)))
}

/// Reproduce the no-Nprime branch of `get_new_nodes`: once an inserted
/// contact is revisited, Z1+ replaces B by the thickness-offset B-dagger
/// point.  Whether that point is a kink is decided only after convergence.
fn settled_contact_position(
    network: &Network,
    a: &Node,
    b: &Node,
    c: &Node,
    pbox: Option<&PeriodicBox>,
    thickness: f64,
) -> Option<Vector3d> {
    let contact = b.contact?;
    let (obstacle_start, obstacle_end, parameter) =
        resolve_obstacle_segment(network, &contact, &b.position, pbox)?;
    let point = obstacle_start + (obstacle_end - obstacle_start) * parameter;
    // When the thickness sphere cuts both incident segments, Z1+'s four
    // Nab/Nbc candidates are active and B is replaced by their B-dagger
    // solution. A tangent contact has only one candidate per side and stays
    // at Nprime.
    let (ab_plus, ab_minus) =
        segment_offset_candidates(&point, thickness, &a.position, &b.position);
    let (bc_plus, bc_minus) =
        segment_offset_candidates(&point, thickness, &b.position, &c.position);
    if std::env::var_os("ENTANGL_Z1PLUS_TRACE").is_some() {
        eprintln!(
            "TRACE settle id={} pos=({:.8},{:.8}) obstacle=({:.8},{:.8}) flags={}/{}/{}/{} db={:+.3e}",
            b.id,
            b.position.x,
            b.position.y,
            point.x,
            point.y,
            ab_plus,
            ab_minus,
            bc_plus,
            bc_minus,
            (b.position - point).norm() - thickness,
        );
    }
    let (weight_a, _, weight_c) = barycentric(&point, &a.position, &b.position, &c.position)?;
    let denominator = 1.0 - weight_a;
    if denominator <= GEOM_EPS {
        return None;
    }
    let bprime = b.position + (c.position - b.position) * (weight_c / denominator);
    let displacement = bprime - b.position;
    let distance = displacement.norm();
    let bdagger = if distance <= GEOM_EPS {
        bprime
    } else {
        bprime - displacement * (thickness / distance).clamp(0.0, 1.0)
    };
    let move_distance = (bdagger - b.position).norm();
    if std::env::var_os("ENTANGL_Z1PLUS_TRACE").is_some() {
        eprintln!("TRACE settle id={} dagger_move={:.9}", b.id, move_distance);
    }
    let endpoint_roundoff = ((b.position - point).norm() - thickness).abs();
    let all_candidates = ab_plus && ab_minus && bc_plus && bc_minus;
    Some(
        if move_distance > 5.0 * thickness
            && (all_candidates
                || (move_distance < 8.0 * thickness
                    && endpoint_roundoff > 5.0 * f32::EPSILON as f64 * thickness))
        {
            bdagger
        } else {
            b.position
        },
    )
}

fn segment_offset_candidates(
    center: &Vector3d,
    radius: f64,
    start: &Vector3d,
    end: &Vector3d,
) -> (bool, bool) {
    let direction = end - start;
    let offset = start - center;
    let length2 = direction.dot(&direction);
    if length2 <= GEOM_EPS {
        return (false, false);
    }
    let projection = -offset.dot(&direction) / length2;
    let closest = start + direction * projection;
    let perpendicular2 = (closest - center).norm_squared();
    if perpendicular2 > radius * radius {
        return (false, false);
    }
    let offset_parameter = ((radius * radius - perpendicular2) / length2).sqrt();
    let plus_parameter = projection + offset_parameter;
    let minus_parameter = projection - offset_parameter;
    let plus = (end - center).norm() > radius && plus_parameter > 0.0 && plus_parameter < 1.0;
    let minus = (start - center).norm() > radius && minus_parameter > 0.0 && minus_parameter < 1.0;
    (plus, minus)
}

fn replacement_shortens(a: &Node, b: &Node, c: &Node, replacement: &[Node]) -> bool {
    let old = (b.position - a.position).norm() + (c.position - b.position).norm();
    let mut new = 0.0;
    let mut previous = a.position;
    for node in replacement {
        new += (node.position - previous).norm();
        previous = node.position;
    }
    new += (c.position - previous).norm();
    new + LENGTH_EPS < old
}

fn best_crossing(
    network: &Network,
    chain_index: usize,
    middle: usize,
    a: &Vector3d,
    b: &Vector3d,
    c: &Vector3d,
    pbox: Option<&PeriodicBox>,
    self_entanglement: bool,
) -> Option<Crossing> {
    let center = (a + b + c) / 3.0;
    let mut best: Option<Crossing> = None;
    for (obstacle_chain_index, chain) in network.chains.iter().enumerate() {
        if obstacle_chain_index == chain_index && !self_entanglement {
            continue;
        }
        for segment_index in 0..chain.nodes.len().saturating_sub(1) {
            if obstacle_chain_index == chain_index
                && segment_index + 2 >= middle
                && segment_index <= middle + 1
            {
                continue;
            }
            let start_node = &chain.nodes[segment_index];
            let end_node = &chain.nodes[segment_index + 1];
            let (start, end) =
                image_segment(&start_node.position, &end_node.position, &center, pbox);
            let Some((point, obstacle_parameter, metric)) =
                segment_triangle_crossing(&start, &end, a, b, c)
            else {
                continue;
            };
            if best
                .as_ref()
                .map_or(true, |current| metric > current.metric)
            {
                best = Some(Crossing {
                    point,
                    metric,
                    obstacle: SegmentId {
                        chain: obstacle_chain_index,
                        start: start_node.id,
                        end: end_node.id,
                    },
                    obstacle_parameter,
                });
            }
        }
    }
    best
}

/// Return the segment/triangle intersection and Z1+'s displacement-selection
/// metric, cos(angle BA, PA), if the intersection lies in both finite objects.
fn segment_triangle_crossing(
    start: &Vector3d,
    end: &Vector3d,
    a: &Vector3d,
    b: &Vector3d,
    c: &Vector3d,
) -> Option<(Vector3d, f64, f64)> {
    let direction = end - start;
    let normal = (b - a).cross(&(c - a));
    let denominator = normal.dot(&direction);
    if denominator.abs() <= GEOM_EPS {
        return None;
    }
    let segment_parameter = normal.dot(&(a - start)) / denominator;
    if !(-GEOM_EPS..=1.0 + GEOM_EPS).contains(&segment_parameter) {
        return None;
    }
    let point = start + direction * segment_parameter;
    let (u, v, w) = barycentric(&point, a, b, c)?;
    if u < -GEOM_EPS || v < -GEOM_EPS || w < -GEOM_EPS {
        return None;
    }
    let ba = a - b;
    let pa = a - point;
    let denominator = ba.norm() * pa.norm();
    if denominator <= GEOM_EPS {
        return None;
    }
    Some((
        point,
        segment_parameter.clamp(0.0, 1.0),
        (ba.dot(&pa) / denominator).clamp(-1.0, 1.0),
    ))
}

fn barycentric(
    point: &Vector3d,
    a: &Vector3d,
    b: &Vector3d,
    c: &Vector3d,
) -> Option<(f64, f64, f64)> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = point - a;
    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() <= GEOM_EPS {
        return None;
    }
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    Some((1.0 - v - w, v, w))
}

fn local_replacement(
    network: &mut Network,
    a: &Node,
    b: &Node,
    c: &Node,
    crossing: Crossing,
    thickness: f64,
) -> Vec<Node> {
    let Some((weight_a, _, weight_c)) =
        barycentric(&crossing.point, &a.position, &b.position, &c.position)
    else {
        return Vec::new();
    };
    let denominator = 1.0 - weight_a;
    if denominator <= GEOM_EPS {
        return Vec::new();
    }
    let bprime = b.position + (c.position - b.position) * (weight_c / denominator);
    let bprime_from_b = bprime - b.position;
    let bprime_distance = bprime_from_b.norm();
    let bdagger = if bprime_distance <= GEOM_EPS {
        bprime
    } else {
        bprime - bprime_from_b * (thickness / bprime_distance).clamp(0.0, 1.0)
    };

    let distance_to_ab = point_segment_distance(&crossing.point, &a.position, &b.position);
    let mut positions = Vec::with_capacity(2);
    if distance_to_ab > thickness + GEOM_EPS {
        let ap = crossing.point - a.position;
        let ba = b.position - a.position;
        let wrap = ap.cross(&ba).cross(&ap);
        if wrap.norm() > GEOM_EPS {
            positions.push((
                crossing.point + wrap.normalize() * thickness,
                NodeKind::Contact,
                Some(Contact {
                    obstacle: crossing.obstacle,
                    obstacle_parameter: crossing.obstacle_parameter,
                    obstacle_s: obstacle_s_from_segment(
                        network,
                        crossing.obstacle,
                        crossing.obstacle_parameter,
                    )
                    .unwrap_or(0.0),
                }),
            ));
        }
    }
    positions.push((bdagger, NodeKind::Ghost, None));
    positions.dedup_by(|left, right| (left.0 - right.0).norm() <= GEOM_EPS);

    let mut lengths = Vec::with_capacity(positions.len() + 1);
    let mut previous = a.position;
    let mut total = 0.0;
    for (position, _, _) in &positions {
        total += (*position - previous).norm();
        lengths.push(total);
        previous = *position;
    }
    total += (c.position - previous).norm();
    if total <= GEOM_EPS {
        return Vec::new();
    }

    let last_index = positions.len().saturating_sub(1);
    positions
        .into_iter()
        .enumerate()
        .map(|(index, (position, kind, contact))| Node {
            id: if index == last_index {
                b.id
            } else {
                network.allocate_id()
            },
            position,
            s: a.s + (c.s - a.s) * lengths[index] / total,
            kind,
            contact,
        })
        .collect()
}

fn point_segment_distance(point: &Vector3d, a: &Vector3d, b: &Vector3d) -> f64 {
    let ab = b - a;
    let denominator = ab.dot(&ab);
    if denominator <= GEOM_EPS {
        return (point - a).norm();
    }
    let t = ((point - a).dot(&ab) / denominator).clamp(0.0, 1.0);
    (point - (a + ab * t)).norm()
}

fn finalize_ghosts(network: &mut Network, pbox: Option<&PeriodicBox>, options: Options) {
    // Collapse near-duplicate contacts attributed to the same obstacle. Z1+'s
    // first best-match pass uses distcrit1 = 5*thickness for this purpose.
    for chain in &mut network.chains {
        let mut index = 1usize;
        while index + 1 < chain.nodes.len() {
            let left = &chain.nodes[index - 1];
            let right = &chain.nodes[index];
            let same_obstacle = left
                .contact
                .zip(right.contact)
                .is_some_and(|(a, b)| a.obstacle == b.obstacle);
            if same_obstacle && (left.position - right.position).norm() < 5.0 * options.thickness {
                chain.nodes.remove(index);
            } else {
                index += 1;
            }
        }
    }

    let distcrit1 = 5.0 * options.thickness;

    // Step 1: best-match every working node against the current path network.
    for chain_index in 0..network.chains.len() {
        if !network.chains[chain_index].movable {
            continue;
        }
        let mut middle = 1usize;
        while middle + 1 < network.chains[chain_index].nodes.len() {
            let node = network.chains[chain_index].nodes[middle].clone();
            let a = network.chains[chain_index].nodes[middle - 1].clone();
            let c = network.chains[chain_index].nodes[middle + 1].clone();
            let binding = closest_obstacle_to_point(
                network,
                chain_index,
                &node.position,
                pbox,
                options.self_entanglement,
            )
            .filter(|(_, distance)| *distance <= distcrit1 + GEOM_EPS);
            let valid_binding = binding.filter(|(contact, _)| {
                node.contact.is_none()
                    || network.chains[contact.obstacle.chain].movable
                    || contact_position(network, *contact, &node.position, pbox)
                        .and_then(|point| {
                            barycentric(&point, &a.position, &node.position, &c.position)
                        })
                        .is_some_and(|(u, v, w)| u >= -GEOM_EPS && v >= -GEOM_EPS && w >= -GEOM_EPS)
            });
            if let Some((contact, _)) = valid_binding {
                let node = &mut network.chains[chain_index].nodes[middle];
                node.kind = NodeKind::Contact;
                node.contact = Some(contact);
            } else {
                let rejected_fixed = node
                    .contact
                    .is_some_and(|contact| !network.chains[contact.obstacle.chain].movable);
                let node = &mut network.chains[chain_index].nodes[middle];
                node.kind = if rejected_fixed {
                    NodeKind::RejectedGhost
                } else {
                    NodeKind::Ghost
                };
                node.contact = None;
            }
            middle += 1;
        }
    }

    // Step 2: prune best-match nodes that disappear on the folded kink path.
    for chain in &mut network.chains {
        if !chain.movable || chain.nodes.len() < 3 {
            continue;
        }
        let boundaries = std::iter::once(0)
            .chain(
                chain
                    .nodes
                    .iter()
                    .enumerate()
                    .skip(1)
                    .take(chain.nodes.len() - 2)
                    .filter(|(_, node)| node.kind == NodeKind::Contact)
                    .map(|(index, _)| index),
            )
            .chain(std::iter::once(chain.nodes.len() - 1))
            .collect::<Vec<_>>();
        for triple in boundaries.windows(3) {
            let middle = triple[1];
            if bend_cos_deviation(
                &chain.nodes[triple[0]],
                &chain.nodes[middle],
                &chain.nodes[triple[2]],
            ) <= KINK_COS_DEVIATION
            {
                chain.nodes[middle].kind = NodeKind::Ghost;
                chain.nodes[middle].contact = None;
            }
        }
    }

    // Step 3: add at most one folded-path kink in every gap. Candidate nodes
    // must remain near another path and the largest chord deviation must be
    // significant on the distcrit1 scale.
    for chain_index in 0..network.chains.len() {
        if !network.chains[chain_index].movable {
            continue;
        }
        let boundaries = std::iter::once(0)
            .chain(
                network.chains[chain_index]
                    .nodes
                    .iter()
                    .enumerate()
                    .skip(1)
                    .take(network.chains[chain_index].nodes.len().saturating_sub(2))
                    .filter(|(_, node)| node.kind == NodeKind::Contact)
                    .map(|(index, _)| index),
            )
            .chain(std::iter::once(network.chains[chain_index].nodes.len() - 1))
            .collect::<Vec<_>>();
        for pair in boundaries.windows(2) {
            let start = network.chains[chain_index].nodes[pair[0]].position;
            let end = network.chains[chain_index].nodes[pair[1]].position;
            let mut best: Option<(usize, f64, Contact)> = None;
            for index in pair[0] + 1..pair[1] {
                if matches!(
                    network.chains[chain_index].nodes[index].kind,
                    NodeKind::Contact | NodeKind::RejectedGhost
                ) {
                    continue;
                }
                let position = network.chains[chain_index].nodes[index].position;
                let Some((contact, distance)) = closest_obstacle_to_point(
                    network,
                    chain_index,
                    &position,
                    pbox,
                    options.self_entanglement,
                ) else {
                    continue;
                };
                if distance > 15.0 * distcrit1 + GEOM_EPS {
                    continue;
                }
                let deviation = point_segment_distance(&position, &start, &end);
                if best
                    .as_ref()
                    .is_none_or(|(_, best_deviation, _)| deviation > *best_deviation)
                {
                    best = Some((index, deviation, contact));
                }
            }
            if let Some((index, deviation, contact)) = best {
                if deviation > 5.0 * distcrit1 + GEOM_EPS {
                    let node = &mut network.chains[chain_index].nodes[index];
                    node.kind = NodeKind::Contact;
                    node.contact = Some(contact);
                }
            }
        }
    }

    for chain in &mut network.chains {
        if !chain.movable || chain.nodes.len() < 3 {
            continue;
        }
        let last = chain.nodes.len() - 1;
        chain.nodes = chain
            .nodes
            .drain(..)
            .enumerate()
            .filter_map(|(index, node)| {
                (index == 0 || index == last || node.kind == NodeKind::Contact).then_some(node)
            })
            .collect();
    }
}

fn contact_position(
    network: &Network,
    contact: Contact,
    reference: &Vector3d,
    pbox: Option<&PeriodicBox>,
) -> Option<Vector3d> {
    let (start, end, parameter) = resolve_obstacle_segment(network, &contact, reference, pbox)?;
    Some(start + (end - start) * parameter)
}

fn closest_obstacle_to_point(
    network: &Network,
    chain_index: usize,
    point: &Vector3d,
    pbox: Option<&PeriodicBox>,
    self_entanglement: bool,
) -> Option<(Contact, f64)> {
    let mut best: Option<(Contact, f64)> = None;
    for (obstacle_chain_index, chain) in network.chains.iter().enumerate() {
        if obstacle_chain_index == chain_index && !self_entanglement {
            continue;
        }
        for pair in chain.nodes.windows(2) {
            let (start, end) = image_segment(&pair[0].position, &pair[1].position, point, pbox);
            let segment = end - start;
            let denominator = segment.dot(&segment);
            let obstacle_parameter = if denominator <= GEOM_EPS {
                0.0
            } else {
                ((point - start).dot(&segment) / denominator).clamp(0.0, 1.0)
            };
            let distance = (point - (start + segment * obstacle_parameter)).norm();
            if best
                .as_ref()
                .is_none_or(|(_, best_distance)| distance < *best_distance)
            {
                best = Some((
                    Contact {
                        obstacle: SegmentId {
                            chain: obstacle_chain_index,
                            start: pair[0].id,
                            end: pair[1].id,
                        },
                        obstacle_parameter,
                        obstacle_s: pair[0].s + (pair[1].s - pair[0].s) * obstacle_parameter,
                    },
                    distance,
                ));
            }
        }
    }
    best
}

fn is_kink(a: &Node, b: &Node, c: &Node) -> bool {
    if b.kind != NodeKind::Contact || b.contact.is_none() {
        return false;
    }
    bend_cos_deviation(a, b, c) > KINK_COS_DEVIATION
}

fn bend_cos_deviation(a: &Node, b: &Node, c: &Node) -> f64 {
    let incoming = b.position - a.position;
    let outgoing = c.position - b.position;
    let denominator = incoming.norm() * outgoing.norm();
    if denominator <= GEOM_EPS {
        0.0
    } else {
        1.0 - incoming.dot(&outgoing) / denominator
    }
}

fn image_segment(
    start: &Vector3d,
    end: &Vector3d,
    target: &Vector3d,
    pbox: Option<&PeriodicBox>,
) -> (Vector3d, Vector3d) {
    let Some(pbox) = pbox else {
        return (*start, *end);
    };
    let midpoint = (start + end) * 0.5;
    let delta = (midpoint - target).cast::<Float>();
    let shift = (pbox.shortest_vector(&delta) - delta).cast::<f64>();
    (start + shift, end + shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: NodeId, x: f64, y: f64, z: f64, s: f64) -> Node {
        Node {
            id,
            position: Vector3d::new(x, y, z),
            s,
            kind: NodeKind::Original,
            contact: None,
        }
    }

    #[test]
    fn recovered_first_benchmark_01_insertion_matches_oracle_trace() {
        let a = node(1, 0.0, 0.0, 0.0, 1.0);
        let b = node(2, -0.999050, -0.043485, 0.0, 2.0);
        let c = node(3, -0.638610, -0.390012, 0.0, 3.0);
        let start = Vector3d::new(-0.689110, -0.035973, -1.0);
        let end = Vector3d::new(-0.689110, -0.035973, 1.0);
        let (point, obstacle_parameter, metric) =
            segment_triangle_crossing(&start, &end, &a.position, &b.position, &c.position).unwrap();
        assert!(metric > 0.9999);

        let mut network = Network {
            chains: Vec::new(),
            next_id: 10,
        };
        let replacement = local_replacement(
            &mut network,
            &a,
            &b,
            &c,
            Crossing {
                point,
                metric,
                obstacle: SegmentId {
                    chain: 1,
                    start: 8,
                    end: 9,
                },
                obstacle_parameter,
            },
            0.002,
        );
        assert_eq!(replacement.len(), 2);
        assert!((replacement[0].position.x - -0.689214).abs() < 2.0e-4);
        assert!((replacement[0].position.y - -0.033976).abs() < 2.0e-4);
        assert!((replacement[1].position.x - -0.991941).abs() < 2.0e-4);
        assert!((replacement[1].position.y - -0.050320).abs() < 2.0e-4);
    }
}
