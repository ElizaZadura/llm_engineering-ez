use std::time::Instant;

fn lcg(seed: u32) -> impl Iterator<Item = u32> {
    let a: u64 = 1664525;
    let c: u64 = 1013904223;
    let m: u64 = 1u64 << 32;
    
    let mut value = seed as u64;
    std::iter::from_fn(move || {
        value = (a * value + c) % m;
        Some(value as u32)
    })
}

fn max_subarray_sum(n: usize, seed: u32, min_val: i32, max_val: i32) -> i64 {
    let range = (max_val - min_val + 1) as u32;
    let mut lcg_gen = lcg(seed);
    
    // Generate random numbers
    let random_numbers: Vec<i32> = (0..n)
        .map(|_| (lcg_gen.next().unwrap() % range) as i32 + min_val)
        .collect();
    
    // Kadane's algorithm for maximum subarray sum
    let mut max_sum = i64::MIN;
    
    for i in 0..n {
        let mut current_sum: i64 = 0;
        for j in i..n {
            current_sum += random_numbers[j] as i64;
            if current_sum > max_sum {
                max_sum = current_sum;
            }
        }
    }
    
    max_sum
}

fn total_max_subarray_sum(n: usize, initial_seed: u32, min_val: i32, max_val: i32) -> i64 {
    let mut total_sum: i64 = 0;
    let mut lcg_gen = lcg(initial_seed);
    
    for _ in 0..20 {
        let seed = lcg_gen.next().unwrap();
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }
    
    total_sum
}

fn main() {
    let n = 10000;
    let initial_seed = 42;
    let min_val = -10;
    let max_val = 10;
    
    let start_time = Instant::now();
    let result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    let end_time = Instant::now();
    
    println!("Total Maximum Subarray Sum (20 runs): {}", result);
    println!("Execution Time: {:.6} seconds", (end_time - start_time).as_secs_f64());
}