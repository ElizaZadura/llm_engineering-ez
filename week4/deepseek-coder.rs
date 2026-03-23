fn lcg(seed: u32, a: u32, c: u32, m: u32) -> impl Iterator<Item = u32> {
    std::iter::successors(Some(seed), move |&val| Some((a * val + c) % m))
}

fn max_subarray_sum(random_numbers: &[i32]) -> i32 {
    let mut max_sum = i32::MIN;
    let mut current_sum;
    for i in 0..random_numbers.len() {
        current_sum = 0;
        for j in i..random_numbers.len() {
            current_sum += random_numbers[j];
            if current_sum > max_sum {
                max_sum = current_sum;
            }
        }
    }
    max_sum
}

fn total_max_subarray_sum(n: usize, initial_seed: u32, min_val: i32, max_val: i32) -> i32 {
    let a = 1664525;
    let c = 1013904223;
    let m = 2u32.pow(32);
    let mut total_sum = 0;
    let mut lcg_gen = lcg(initial_seed, a, c, m);
    for _ in 0..20 {
        let seed = lcg_gen.next().unwrap();
        let mut random_numbers = Vec::new();
        for _ in 0..n {
            let num = (seed as u32) % (max_val - min_val + 1) as u32 + min_val;
            random_numbers.push(num as i32);
        }
        total_sum += max_subarray_sum(&random_numbers);
    }
    total_sum
}

fn main() {
    let n = 10000;
    let initial_seed = 42;
    let min_val = -10;
    let max_val = 10;

    let start_time = std::time::Instant::now();
    let result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    let end_time = std::time::Instant::now();

    println!("Total Maximum Subarray Sum (20 runs): {}", result);
    println!("Execution Time: {:.6} seconds", end_time.duration_since(start_time).as_secs_f64());
}