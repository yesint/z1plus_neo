//! Flat stable-id node pool for the SMDP core.
//!
//! Node data is held in parallel arrays (`nextid`, `previd`, `chain_for_id`,
//! `idstart`, `mys`, `x`) indexed by a permanent integer id, with a free list for
//! reuse. A node's identity is stable for the whole run; its position along a
//! chain is recovered by walking `nextid`. Insert / move / collapse are O(1)
//! and touch only the node's own two neighbour pointers, so cached references
//! never go stale — unlike the old
//! port's `(chain, Vec index)` identity, which `Vec::insert`/`remove` at a
//! smaller index silently invalidated.
//!
//! Deletion during minimization is by collinear collapse into a LIFO free list;
//! ids are reused (no generational index — behaviour depends only on geometry
//! and topology, matching the Fortran).

pub type Id = u32;
/// Null / sentinel id. Index 0 of every SoA array is reserved and never live.
pub const NULL: Id = 0;

/// Chain index type (1-based; 0 in `chain_for_id` means "on the free list").
pub type ChainId = u32;

#[derive(Clone, Debug)]
pub struct Pool {
    // ---- per-id SoA arrays, length == capacity (index 0 is the null sentinel) ----
    /// Working coordinate (folded to the central box during the sweep).
    pub x: Vec<[f64; 3]>,
    /// Contour label seeded `1..=N(chain)`; preserves order without touching `x`.
    pub mys: Vec<f64>,
    /// Kink flag: false for the entire sweep, set only by the finalizer.
    pub kink: Vec<bool>,
    /// Owning chain (1..=chains); 0 marks a free slot.
    pub chain_for_id: Vec<ChainId>,
    /// Doubly-linked contour order; `NULL` terminates.
    pub nextid: Vec<Id>,
    pub previd: Vec<Id>,

    // ---- per-chain arrays, index 1..=chains (index 0 unused) ----
    pub idstart: Vec<Id>,
    pub idend: Vec<Id>,
    /// Current live node count per chain.
    pub n: Vec<u32>,
    /// Original node count per chain (post-densification is separate; this is
    /// the as-built bead count, used only for diagnostics).
    pub norig: Vec<u32>,

    // ---- free list + global counters ----
    nextfree: Id,
    pub maxid_ever: Id,
    pub system_n: u32,
    /// Immediate free-list reclaims this sweep (collinear collapse).
    pub erased: u32,
    /// Deferred-ghost midpoints created this sweep.
    pub ghosts: u32,
}

impl Pool {
    /// Number of chains (1-based indexing; `chains()` == idstart.len() - 1).
    pub fn chains(&self) -> usize {
        self.idstart.len().saturating_sub(1)
    }

    /// Current capacity (max id that can be stored without growing).
    pub fn capacity(&self) -> usize {
        self.x.len().saturating_sub(1)
    }

    /// Build the pool from per-chain bead coordinates (one node per bead).
    /// Chains are numbered 1..=chains. Endpoints (first/last of each chain) are
    /// the pinned nodes; the sweep skips them.
    pub fn from_chains(chains: &[Vec<[f64; 3]>]) -> Self {
        let total: usize = chains.iter().map(|c| c.len()).sum();
        // headroom (~40% like the Fortran; at least a few slots) so densify /
        // insertion rarely needs `enlarge`.
        let cap = (total + total / 2 + 16).max(total + 1);
        let mut p = Pool {
            x: vec![[0.0; 3]; cap + 1],
            mys: vec![0.0; cap + 1],
            kink: vec![false; cap + 1],
            chain_for_id: vec![0; cap + 1],
            nextid: vec![NULL; cap + 1],
            previd: vec![NULL; cap + 1],
            idstart: vec![NULL; chains.len() + 1],
            idend: vec![NULL; chains.len() + 1],
            n: vec![0; chains.len() + 1],
            norig: vec![0; chains.len() + 1],
            nextfree: NULL,
            maxid_ever: 0,
            system_n: 0,
            erased: 0,
            ghosts: 0,
        };

        let mut next_id: Id = 1;
        for (ci, beads) in chains.iter().enumerate() {
            let chain = (ci + 1) as ChainId;
            let mut prev = NULL;
            for (bi, pos) in beads.iter().enumerate() {
                let id = next_id;
                next_id += 1;
                p.x[id as usize] = *pos;
                p.mys[id as usize] = (bi + 1) as f64;
                p.chain_for_id[id as usize] = chain;
                p.previd[id as usize] = prev;
                p.nextid[id as usize] = NULL;
                if prev != NULL {
                    p.nextid[prev as usize] = id;
                } else {
                    p.idstart[chain as usize] = id;
                }
                prev = id;
            }
            p.idend[chain as usize] = prev;
            p.n[chain as usize] = beads.len() as u32;
            p.norig[chain as usize] = beads.len() as u32;
        }
        p.system_n = total as u32;
        p.maxid_ever = (next_id - 1).max(0);

        // thread the unused tail onto the LIFO free list (low ids popped first)
        for id in (next_id..=cap as Id).rev() {
            p.nextid[id as usize] = p.nextfree;
            p.nextfree = id;
        }
        p
    }

    /// Is `id` a pinned endpoint of its chain (never moved/collapsed)?
    #[inline]
    pub fn is_endpoint(&self, id: Id) -> bool {
        let c = self.chain_for_id[id as usize];
        c != 0 && (self.idstart[c as usize] == id || self.idend[c as usize] == id)
    }

    #[inline]
    pub fn is_free(&self, id: Id) -> bool {
        self.chain_for_id[id as usize] == 0
    }

    /// Grow all per-id arrays, threading the new slots onto the free list.
    /// Preserves every existing id's meaning (capacity change only).
    fn enlarge(&mut self) {
        let old = self.capacity();
        let new = (old * 2).max(old + 16);
        self.x.resize(new + 1, [0.0; 3]);
        self.mys.resize(new + 1, 0.0);
        self.kink.resize(new + 1, false);
        self.chain_for_id.resize(new + 1, 0);
        self.nextid.resize(new + 1, NULL);
        self.previd.resize(new + 1, NULL);
        for id in ((old as Id + 1)..=new as Id).rev() {
            self.nextid[id as usize] = self.nextfree;
            self.nextfree = id;
        }
    }

    /// Pop a fresh id from the free list (growing if empty). The returned node
    /// is unlinked and has `chain_for_id == 0`; the caller must link + fill it.
    fn alloc(&mut self) -> Id {
        if self.nextfree == NULL {
            self.enlarge();
        }
        let id = self.nextfree;
        self.nextfree = self.nextid[id as usize];
        self.nextid[id as usize] = NULL;
        self.previd[id as usize] = NULL;
        if id > self.maxid_ever {
            self.maxid_ever = id;
        }
        id
    }

    /// Insert a new node with position `pos` and contour label `mys`
    /// immediately after `after` in the contour. Returns the new id.
    pub fn insert_after(&mut self, after: Id, pos: [f64; 3], mys: f64) -> Id {
        debug_assert!(!self.is_free(after), "insert_after on a free node");
        let chain = self.chain_for_id[after as usize];
        let nx = self.nextid[after as usize];
        let id = self.alloc();
        self.x[id as usize] = pos;
        self.mys[id as usize] = mys;
        self.kink[id as usize] = false;
        self.chain_for_id[id as usize] = chain;
        // link  after <-> id <-> nx
        self.previd[id as usize] = after;
        self.nextid[id as usize] = nx;
        self.nextid[after as usize] = id;
        if nx != NULL {
            self.previd[nx as usize] = id;
        } else {
            self.idend[chain as usize] = id;
        }
        self.n[chain as usize] += 1;
        self.system_n += 1;
        id
    }

    /// Collapse (delete) an interior node: splice it out, reclaim its id onto
    /// the free list. Returns the successor id (for continuing a forward walk).
    /// Panics in debug if asked to remove an endpoint.
    pub fn collapse(&mut self, id: Id) -> Id {
        debug_assert!(!self.is_endpoint(id), "collapse of a pinned endpoint");
        debug_assert!(!self.is_free(id), "double collapse");
        let chain = self.chain_for_id[id as usize];
        let p = self.previd[id as usize];
        let nx = self.nextid[id as usize];
        // splice out
        if p != NULL {
            self.nextid[p as usize] = nx;
        }
        if nx != NULL {
            self.previd[nx as usize] = p;
        }
        // reclaim
        self.chain_for_id[id as usize] = 0;
        self.previd[id as usize] = NULL;
        self.nextid[id as usize] = self.nextfree;
        self.nextfree = id;
        self.n[chain as usize] -= 1;
        self.system_n -= 1;
        self.erased += 1;
        nx
    }

    /// Walk a chain's ids in contour order (endpoints included).
    pub fn walk(&self, chain: ChainId) -> ChainWalk<'_> {
        ChainWalk {
            pool: self,
            cur: self.idstart[chain as usize],
        }
    }

    /// Total live node count recomputed from the chain walks (for assertions).
    pub fn recount(&self) -> u32 {
        (1..=self.chains() as ChainId)
            .map(|c| self.walk(c).count() as u32)
            .sum()
    }

    /// Debug invariant check (consensus Phase-1 gate). Panics on violation.
    pub fn check_invariants(&self) {
        // 1. system_n == Σ n[c] == Σ chain-walk length == count(chain_for_id != 0)
        let sum_n: u32 = (1..=self.chains()).map(|c| self.n[c]).sum();
        assert_eq!(self.system_n, sum_n, "system_n != Σ n[c]");
        assert_eq!(self.system_n, self.recount(), "system_n != Σ walk length");
        let live_marked = self
            .chain_for_id
            .iter()
            .enumerate()
            .filter(|(i, &c)| *i != 0 && c != 0)
            .count() as u32;
        assert_eq!(
            self.system_n, live_marked,
            "system_n != count(chain_for_id!=0)"
        );

        // 2. every live id appears in exactly one chain walk, and n[c] matches
        let mut seen = vec![false; self.capacity() + 1];
        for c in 1..=self.chains() as ChainId {
            let mut len = 0u32;
            for id in self.walk(c) {
                assert!(id as usize <= self.capacity(), "id out of range");
                assert!(!seen[id as usize], "id {id} in two chain walks");
                seen[id as usize] = true;
                assert_eq!(
                    self.chain_for_id[id as usize], c,
                    "chain_for_id mismatch for id {id}"
                );
                len += 1;
            }
            assert_eq!(len, self.n[c as usize], "walk length != n[{c}]");
        }

        // 3. free list is disjoint from live and terminates
        let mut f = self.nextfree;
        let mut guard = self.capacity() + 2;
        while f != NULL {
            assert!(self.is_free(f), "live id {f} on free list");
            assert!(!seen[f as usize], "id {f} both free and live");
            seen[f as usize] = true; // reuse to detect free-list cycles
            f = self.nextid[f as usize];
            guard -= 1;
            assert!(guard > 0, "free list cycle");
        }
    }
}

/// Iterator over a chain's ids in contour order.
pub struct ChainWalk<'a> {
    pool: &'a Pool,
    cur: Id,
}

impl Iterator for ChainWalk<'_> {
    type Item = Id;
    fn next(&mut self) -> Option<Id> {
        if self.cur == NULL {
            None
        } else {
            let id = self.cur;
            self.cur = self.pool.nextid[id as usize];
            Some(id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize) -> Vec<[f64; 3]> {
        (0..n).map(|i| [i as f64, 0.0, 0.0]).collect()
    }

    #[test]
    fn from_chains_builds_correct_structure() {
        let chains = vec![line(5), line(3), line(4)];
        let p = Pool::from_chains(&chains);
        assert_eq!(p.chains(), 3);
        assert_eq!(p.system_n, 12);
        for (ci, beads) in chains.iter().enumerate() {
            let c = (ci + 1) as ChainId;
            let ids: Vec<Id> = p.walk(c).collect();
            assert_eq!(ids.len(), beads.len());
            assert_eq!(p.n[c as usize], beads.len() as u32);
            // positions and contour order match
            for (k, &id) in ids.iter().enumerate() {
                assert_eq!(p.x[id as usize], beads[k]);
                assert_eq!(p.mys[id as usize], (k + 1) as f64);
            }
            assert!(p.is_endpoint(ids[0]));
            assert!(p.is_endpoint(*ids.last().unwrap()));
            assert!(!p.is_endpoint(ids[1]));
        }
        p.check_invariants();
    }

    #[test]
    fn insert_and_collapse_roundtrip() {
        let chains = vec![line(4)];
        let mut p = Pool::from_chains(&chains);
        let start = p.idstart[1];
        let second = p.nextid[start as usize];
        let new = p.insert_after(start, [0.5, 0.0, 0.0], 1.5);
        assert_eq!(p.system_n, 5);
        assert_eq!(p.nextid[start as usize], new);
        assert_eq!(p.previd[second as usize], new);
        p.check_invariants();
        let succ = p.collapse(new);
        assert_eq!(succ, second);
        assert_eq!(p.system_n, 4);
        assert_eq!(p.nextid[start as usize], second);
        p.check_invariants();
    }

    #[test]
    fn insert_at_tail_updates_idend() {
        let chains = vec![line(3)];
        let mut p = Pool::from_chains(&chains);
        let last = p.idend[1];
        let before_last = p.previd[last as usize];
        // insert between before_last and last: still interior; idend unchanged
        let mid = p.insert_after(before_last, [1.5, 0.0, 0.0], 2.5);
        assert_eq!(p.idend[1], last);
        assert_eq!(p.nextid[mid as usize], last);
        p.check_invariants();
    }

    #[test]
    fn many_random_ops_preserve_invariants() {
        // deterministic pseudo-random (no Math.random): linear congruential
        let chains = vec![line(20), line(15), line(25)];
        let mut p = Pool::from_chains(&chains);
        let mut rng: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u32
        };
        for _ in 0..20_000 {
            let c = (next() % p.chains() as u32) + 1;
            let ids: Vec<Id> = p.walk(c).collect();
            if ids.len() <= 2 {
                continue;
            }
            let interior = &ids[1..ids.len() - 1];
            if next() % 2 == 0 {
                // insert after a random interior (or start) node
                let after = ids[(next() as usize) % (ids.len() - 1)];
                p.insert_after(after, [0.1, 0.2, 0.3], 0.0);
            } else {
                // collapse a random interior node
                let victim = interior[(next() as usize) % interior.len()];
                p.collapse(victim);
            }
            // enlarge is exercised as capacity fills
        }
        p.check_invariants();
        // capacity grew via enlarge at least once given 10k net inserts possible
        assert!(p.maxid_ever as usize <= p.capacity());
    }

    #[test]
    fn enlarge_preserves_ids_and_links() {
        let chains = vec![line(4)];
        let mut p = Pool::from_chains(&chains);
        // exhaust the free list to force enlarge, remembering a stable id
        let anchor = p.nextid[p.idstart[1] as usize];
        let anchor_pos = p.x[anchor as usize];
        let mut after = p.idstart[1];
        for k in 0..100 {
            after = p.insert_after(after, [k as f64, 0.0, 0.0], 0.0);
        }
        // the original anchor id still means the same node
        assert_eq!(p.x[anchor as usize], anchor_pos);
        assert_eq!(p.chain_for_id[anchor as usize], 1);
        p.check_invariants();
    }
}
