1. not a pip-installable package
2. library not in maintainance: 
   - https://github.com/DeepGraphLearning/torchdrug does not support Python>3.10 (target: 3.10-3.14)
   - manual forcing installation works yet risky: `pip install --ignore-requires-python ...`
   - possible to get this updated? or get a copy of called code into diffpack?
   - a clone of torchdrug: `/Users/yyy/Documents/protein_design/torchdrug`
3. MPS GPU device not supported
4. numpy 2x not supported (minimum 1.26.4 for legacy compatibility (if possible))
5. takes too long to finish a task (runtime optimization needed?)
6. not PyG-ready ()
7. deprecated Torch callings
8. failed runs : 
   - (macos, `clang++: error: unsupported option '-fopenmp'`)
    ```
    python script/inference.py -c config/inference_confidence.yaml \
        --seed 2023 \
        --output_dir output \
        --pdb_files 10mh_A.pdb \
        --center_residues A:72 A:155 \
        --repack_radius 10
    ```
9. no test cases or suite. need comprehensive unit tests and intergation tests
10. no benchmarks accross CPU and GPU
11. docs out-of-dated