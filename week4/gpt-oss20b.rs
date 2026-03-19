use std::time::Instant;

const N: usize = 10000;
const INITIAL_SEED: u32 = 42;
const MIN_VAL: i32 = -10;
const MAX_VAL: i32 = 10;

fn max_subarray_sum(n: usize, seed: u32, min_val: i32, max_val: i32) -> i32 {
    let mut state = seed;
    let range = (max_val - min_val + 1) as u32;
    let mut curr_sum: i32 = 0;
    let mut best: i32 = i32::MIN;

    for _ in 0..n {
        state = state
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        let val = ((state % range) as i32) + min_val;

        if curr_sum > 0 {
            curr_sum += val;
        } else {
            curr_sum = val;
        }

        if curr_sum > best {
            best = curr_sum;
        }
    }

    best
}

fn total_max_subarray_sum(n: usize, initial_seed: u32, min_val: i32, max_val: i32) -> i32 {
    let mut total_sum = 0i32;
    let mut state = initial_seed;

    for _ in 0..20 {
        state = state
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        let seed = state;
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }

    total_sum
}

fn main() {
    let start = Instant::now();
    let result = total_max_subarray_sum(N, INITIAL_SEED, MIN_VAL, MAX_VAL);
    let elapsed = start.elapsed().as_secs_f64();

    println!("Total Maximum Subarray Sum (20 runs): {}", result);
    println!("Execution Time: {:.6} seconds", elapsed);
}