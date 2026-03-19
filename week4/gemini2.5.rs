use std::time::Instant;

// A Linear Congruential Generator implementation.
struct Lcg {
    state: u32,
}

impl Lcg {
    #[inline(always)]
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }
}

impl Iterator for Lcg {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: u64 = 1 << 32;

        // Perform calculation in u64 to match Python's arbitrary-precision integer
        // behavior for the intermediate `a * value + c` step before the modulo.
        self.state = ((A * self.state as u64 + C) % M) as u32;
        Some(self.state)
    }
}

// Finds the maximum subarray sum for a sequence of pseudo-random numbers.
// This uses Kadane's algorithm for O(n) performance, a significant
// improvement over the O(n^2) approach in the original Python code,
// while producing an identical result.
fn max_subarray_sum(n: usize, seed: u32, min_val: i64, max_val: i64) -> i64 {
    // This function assumes n > 0 based on the problem's parameters.
    let mut lcg = Lcg::new(seed);
    let range = max_val - min_val + 1;

    // Generate the first number to initialize the state.
    let first_num = (lcg.next().unwrap() as i64 % range) + min_val;
    let mut max_so_far = first_num;
    let mut max_ending_here = first_num;

    // Process the remaining n-1 numbers without storing them in a collection.
    for _ in 1..n {
        let num = (lcg.next().unwrap() as i64 % range) + min_val;
        max_ending_here = num.max(max_ending_here + num);
        max_so_far = max_so_far.max(max_ending_here);
    }

    max_so_far
}

// Calculates the total sum of max_subarray_sum over 20 runs with different seeds.
fn total_max_subarray_sum(n: usize, initial_seed: u32, min_val: i64, max_val: i64) -> i64 {
    let mut lcg_seeds = Lcg::new(initial_seed);
    let mut total_sum: i64 = 0;

    for _ in 0..20 {
        let seed = lcg_seeds.next().unwrap();
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }
    total_sum
}

fn main() {
    const N: usize = 10000;
    const INITIAL_SEED: u32 = 42;
    const MIN_VAL: i64 = -10;
    const MAX_VAL: i64 = 10;

    let start_time = Instant::now();
    let result = total_max_subarray_sum(N, INITIAL_SEED, MIN_VAL, MAX_VAL);
    let duration = start_time.elapsed();

    println!("Total Maximum Subarray Sum (20 runs): {}", result);
    println!("Execution Time: {:.6f} seconds", duration.as_secs_f64());
}