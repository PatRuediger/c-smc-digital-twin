# C-SMC Synthetic Data Generator for Vision-Based Inline Inspection

A process-informed, photorealistic simulation pipeline that generates synthetic training data for machine learning models in carbon-fiber Sheet Moulding Compound (C-SMC) manufacturing. The pipeline simulates rectangular SMC strips falling from an outlet onto a moving conveyor belt, renders polarization images at four analyzer angles, computes Angle of Linear Polarization (AoLP) composites, and produces fully annotated datasets for object detection and density estimation.

## Paper

> **A Process-Informed, Photorealistic Synthetic Data Generator for Vision-Based Inline Inspection of C-SMC Strip Deposition**
>
> Patrick Flore, Fabian Roeder, Kevin Chen, Andreas Gebhard
>
> *Composites Part A: Applied Science and Manufacturing* -- under review
>
> Preprint available on request.

## Architecture

```
config.json
    |
    v
run_simulation.py  -->  blender -b  -->  stripGen_Ana_automate.py
   (batch runner;                          (thin CLI entry point;
    one subprocess                         parses --seed --config,
    per seed)                              calls run_full_pipeline)
                                                     |
                                                     v
                                           simulation/ package
                                           +------------------------------+
                                           |  pipeline.py                 |
                                           |   StripDroppingSimulation    |
                                           |   (assembles all modules)    |
                                           |                              |
                                           |  config.py    database.py    |
                                           |  scene.py     physics.py     |
                                           |  density.py   rendering.py   |
                                           |  polarization.py             |
                                           |  annotation.py               |
                                           +------------------------------+
                                                     |
                          +--------------------------+-------------------------+
                          v                          v                          v
                  Rendered .png images        YOLO OBB .txt             SQLite database
                  (polarization series        annotation files          (configs + density
                   + AoLP composite)                                     + image paths)
                          |
              +-----------+----------------+
              v                            v
  AnnotationPreprocessing.ipynb     analysis_nb.ipynb
              |
              v
  YOLO training (yolo/)
  CSRNet training (csrnet/)
```

## Quickstart

### Prerequisites

- **Blender 4.0 or newer** -- download from [blender.org](https://www.blender.org/download/)
- **Python 3.10+** (external runner only; Blender ships its own Python)
- No additional Python packages required for the simulation itself

### Installation

```bash
git clone https://github.com/PatRuediger/c-smc-digital-twin.git
cd c-smc-digital-twin
```

### Configure

Copy the example config and edit it for your environment:

```bash
cp config.example.json config.json
```

Set at minimum:

- `output_paths.render_output_path` -- absolute path where rendered images will be written
- `output_paths.db_output_path` -- absolute path where the SQLite database will be written
- `simulation_run.seeds` -- list of integer seeds to simulate

### Run

```bash
python run_simulation.py config.json
```

This auto-detects Blender (or reads the `BLENDER_PATH` environment variable) and spawns one `blender -b` subprocess per seed. Each process runs the full pipeline: scene setup, rigid-body physics bake, polarization rendering, AoLP computation, YOLO OBB annotation, density calculation, and database write.

To override the Blender path:

```bash
BLENDER_PATH=/path/to/blender python run_simulation.py config.json
```

## Repository Layout

```
c-smc-digital-twin/
+-- simulation/                     # Blender-side Python package
|   +-- config.py                   # SimulationConfig + StripData
|   +-- scene.py                    # Scene setup, belt, spawn zone, strip creation
|   +-- physics.py                  # Spawn scheduling, collision checking
|   +-- density.py                  # 3D volume + 2D shadow density calculation
|   +-- polarization.py             # Shader node groups, material assignment
|   +-- annotation.py               # YOLO OBB annotation generation
|   +-- rendering.py                # Polarization rendering, AoLP computation
|   +-- database.py                 # DatabaseManager (SQLite)
|   +-- pipeline.py                 # StripDroppingSimulation + run_full_pipeline()
+-- stripGen_Ana_automate.py        # Thin entry point: CLI args -> run_full_pipeline()
+-- run_simulation.py               # Batch runner (external Python)
+-- generate_configs.py             # Grid-sweep config generator
+-- generate_batch_scripts.py       # SLURM cluster submit script generator
+-- config.example.json             # Template config with placeholder paths
+-- StripsGen_outLetSimulation_init.blend  # Base Blender scene
+-- yolo/                           # YOLO model training and evaluation
+-- csrnet/                         # CSRNet density model training and evaluation
+-- methods_comparison/             # Quantitative method comparison pipeline
|   +-- watershed.py                # Watershed baseline
|   +-- topology_ttk.py             # Morse-Smale complex centroid counting (TTK)
|   +-- csrnet_infer.py             # CSRNet inference wrapper
|   +-- fusion_head.py              # Late-fusion MLP ablation
|   +-- run_eval.py                 # Evaluation harness
|   +-- figures.py                  # Publication figure generation
|   +-- bootstrap_ci.py             # Bootstrap confidence intervals
|   +-- real_frames/                # Real production AoLP frames for evaluation
|   +-- tests/                      # Unit tests
+-- AnnotationPreprocessing.ipynb   # DB output -> YOLO annotation format
+-- analysis_nb.ipynb               # Post-hoc analysis of simulation results
```

## Outputs

A completed batch run produces:

- **Rendered images**: polarization series (4 angles) and AoLP composite per seed/frame
- **SQLite database**: one row per (seed, frame) with full config, density values, and image paths
- **YOLO OBB annotations**: `.txt` files alongside rendered images
- **Config backup**: timestamped copy of `config.json` for reproducibility

## Citation

If you use this code in your research, please cite:

```bibtex
@article{flore2025csmc_synth_data,
  title={A Process-Informed, Photorealistic Synthetic Data Generator for Vision-Based Inline Inspection of {C-SMC} Strip Deposition},
  author={Flore, Patrick and R{\"o}der, Fabian and Chen, Kevin and Gebhard, Andreas},
  journal={Composites Part A: Applied Science and Manufacturing},
  year={2025},
  note={Under review}
}
```

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.
