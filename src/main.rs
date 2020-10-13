use petgraph::{Graph, Undirected, graph::NodeIndex, visit::EdgeRef};
use ndarray::{Array1, Zip};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;
use sprs::TriMat;
use sprs::errors::SprsError;
use sprs_ldl::Ldl;


fn main() -> Result<(), SprsError> {
    // Seeded for determinism
    let mut rng = Isaac64Rng::seed_from_u64(10_000);

    let n = 2_000;
    let mut graph = petgraph::Graph::with_capacity(n, 20_000);

    for _ in 0..n {
        graph.add_node(());
    }

    let probability = 2.0 * (19_900.) / ((n * (n - 1)) as f64);
    assert!(probability < 1.0);
    for v in 0..n {
        for u in (v + 1)..n {
            if rng.gen_range(0.0, 1.0) < probability {
                let w = 2.0 * ((rng.gen_range(0, 2) - 1) as f64);
                graph.add_edge(NodeIndex::new(v), NodeIndex::new(u), w);
            }
        }
    }
 

    Uniform::new(0.0, 1.0);

    let new_x = Array1::random_using((n,), Uniform::new(0., 3.14159), &mut rng);
    let old_x = Array1::random_using((n,), Uniform::new(0., 3.14159), &mut rng);

    let (rhs1, jacobian1) = compute_value_and_jacobian(&new_x, &old_x, &graph);
    let jacobian1 = jacobian1.to_csc();

    let new_x = Array1::random_using((n,), Uniform::new(0., 3.14159), &mut rng);
    let old_x = Array1::random_using((n,), Uniform::new(0., 3.14159), &mut rng);

    let (rhs2, jacobian2) = compute_value_and_jacobian(&new_x, &old_x, &graph);
    let jacobian2 = jacobian2.to_csc();

    let new_x = Array1::random_using((n,), Uniform::new(0., 3.14159), &mut rng);
    let old_x = Array1::random_using((n,), Uniform::new(0., 3.14159), &mut rng);

    let (rhs3, jacobian3) = compute_value_and_jacobian(&new_x, &old_x, &graph);
    let jacobian3 = jacobian3.to_csc();

    // Compute ref solutions
    let mut ldl = Ldl::new()
        .fill_in_reduction(sprs::FillInReduction::CAMDSuiteSparse)
        .check_symmetry(sprs::DontCheckSymmetry)
        .numeric(jacobian1.view())?;
    let ref1 = ldl.solve(&rhs1.view().to_slice().unwrap());

    ldl.update(jacobian2.view())?;
    let ref2 = ldl.solve(&rhs2.view().to_slice().unwrap());

    ldl.update(jacobian3.view())?;
    let ref3 = ldl.solve(&rhs3.view().to_slice().unwrap());

    let reductions = [
        sprs::FillInReduction::NoReduction,
        sprs::FillInReduction::ReverseCuthillMcKee,
        sprs::FillInReduction::CAMDSuiteSparse,
    ];
    for red in &reductions {
        let tstart = std::time::Instant::now();
        let mut ldl = Ldl::new()
            .fill_in_reduction(*red)
            .check_symmetry(sprs::DontCheckSymmetry)
            .numeric(jacobian1.view())?;
        let solution1 = ldl.solve(&rhs1.view().to_slice().unwrap());

        ldl.update(jacobian2.view())?;
        let solution2 = ldl.solve(&rhs2.view().to_slice().unwrap());

        ldl.update(jacobian3.view())?;
        let solution3 = ldl.solve(&rhs3.view().to_slice().unwrap());

        let tstop = std::time::Instant::now();

        println!(
            "LdlNumeric (rust, {:?}) took {}ms to solve 3 sparse systems",
            *red,
            (tstop - tstart).as_millis(),
        );
        let mut sum = 0.0;
        for (x, y) in solution1.iter().zip(&ref1) {
            sum += (*x - *y).powi(2);
        }
        for (x, y) in solution2.iter().zip(&ref2) {
            sum += (*x - *y).powi(2);
        }
        for (x, y) in solution3.iter().zip(&ref3) {
            sum += (*x - *y).powi(2);
        }
        println!("L2 diff to ref {}", sum.sqrt());
    }
    for red in &reductions {
        let tstart = std::time::Instant::now();
        let mut ldl = Ldl::new()
            .fill_in_reduction(*red)
            .check_symmetry(sprs::DontCheckSymmetry)
            .numeric_c(jacobian1.view())?;
        let solution1 = ldl.solve(&rhs1.view().to_slice().unwrap());

        ldl.update(jacobian2.view())?;
        let solution2 = ldl.solve(&rhs2.view().to_slice().unwrap());

        ldl.update(jacobian3.view())?;
        let solution3 = ldl.solve(&rhs3.view().to_slice().unwrap());

        let tstop = std::time::Instant::now();

        println!(
            "LdlNumeric (C, {:?}) took {}ms to solve 3 sparse systems",
            *red,
            (tstop - tstart).as_millis(),
        );
        let mut sum = 0.0;
        for (x, y) in solution1.iter().zip(&ref1) {
            sum += (*x - *y).powi(2);
        }
        for (x, y) in solution2.iter().zip(&ref2) {
            sum += (*x - *y).powi(2);
        }
        for (x, y) in solution3.iter().zip(&ref3) {
            sum += (*x - *y).powi(2);
        }
        println!("L2 diff to ref {}", sum.sqrt());
    }

    Ok(())
}

fn compute_value_and_jacobian(
    new_x: &Array1<f64>,
    old_x: &Array1<f64>,
    graph: &Graph<(), f64, Undirected, usize>,
) -> (Array1<f64>, TriMat<f64>) {
    let n = graph.node_count();

    let mut jacobian = TriMat::with_capacity((n, n), 2 * graph.edge_count() + n);
    let mut vec_y = Array1::zeros(n);

    Zip::indexed(new_x)
        .and(old_x)
        .and(&mut vec_y)
        .apply(
            |i: usize, new_x_val: &f64, cur_x_val: &f64, y: &mut f64| {
                let mut delta_xi = 0.0;
                let mut ddelta_xi_dxi = 0.0;
                // for each edge attached to the the current node:
                for e in graph.edges(NodeIndex::new(i)) {
                    // Ensures that even if this edge considers i to be the target
                    // (which I believe is impossible) we draw from the ohter side
                    // of the edge.
                    let j = e.target().index();
                    let j = if j != i { j } else { e.source().index() };
                    assert!(i != j);

                    // for each edge we add A_c * J_ij * c(pi * (x[i] - x[j]))
                    delta_xi += e.weight() * (*new_x_val - new_x[j]).sin();

                    ddelta_xi_dxi += e.weight() * (*new_x_val - new_x[j]).cos();

                    jacobian.add_triplet(
                        i,
                        j,
                        -e.weight() * (*new_x_val - new_x[j]).cos()
                    );
                }

                *y = -*new_x_val + *cur_x_val + delta_xi;

                let dxi_dxi = -1.0 + ddelta_xi_dxi;
                jacobian.add_triplet(i, i, dxi_dxi);
            },
        );

    (vec_y, jacobian)
}
