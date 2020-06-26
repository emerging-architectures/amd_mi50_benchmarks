---

title: Initial Assessment of the AMD MI50 GPGPUs for Scientific and Machine Learning Applications
authors: Keith Obenschain, Doug Schwer, and Alisha Sharma
affiliation: Laboratories for Computational Physics and Fluid Dynamics<br>
             Naval Research Laboratory

---

<div class="flow-text">
	<p>
	Competition in the High-Performance Computing GPGPU market has emerged with GPGPUs from Advanced Micro Devices (AMD) and Intel targeting future Exascale class systems.
	The new AMD Radeon Instinct MI50 hints at the capabilities of AMD’s future GPUs.
	</p>
	<p>
	This study takes a first look at the MI50 performance on characteristic scientific and machine learning applications.
	</p>
</div>

<div class="toc" markdown="1">


### Quick Links

1. [The Poster](#the-poster)
2. Presentation (Video) - Coming Soon
3. [Supplemental Data: Reproducing the Benchmarks](#supplemental-data-reproducing-the-benchmarks)
4. Supplemental Data: Rotating Detonation Engine - Coming Soon
</div>


### The Poster

[![ISC-HPC 20 Poster][poster_preview]][poster]

[poster_preview]: /assets/images/poster.png
[poster]: /assets/images/poster.pdf


### Supplemental Data: Reproducing the Benchmarks

We evaluated four reference systems representing a variety of modern consumer- and professional-grade hardware:
1. a **GTX 1080Ti** system with PCIe gen 3x16 GPU-GPU IC,
2. a **RTX 2080Ti** system with no GPU-GPU communication,
3. a **Tesla V100** system with NVLink, and
4. a **AMD Radeon Instinct MI50** system with two xGMI hives.

All systems are use the single-root, dual-socket SuperMicro SYS-4029GP-TRT2 system.
System details, diagrams, and benchmarking results can be found in the poster.

We ran two case studies that represent common workloads we run in our lab:
(a) a GPU-optimized rotating detonation engine simulation, and
(b) a compute-heavy deep learning training task.
Implementation and methodology details are described below.


#### Running the Machine Learning Benchmarks

Our machine learning benchmarks were run using TensorFlow 1.15 and [TensorFlow 1.x CNN benchmarks][tf_cnn_benchmarks].
While the tensorflow benchmarks are no longer updated for TensorFlow 2.x, they have been optimized for TensorFlow 1.15, making this a useful and replicable task for comparing GPGPU performance.

Because our tests are run on a single node, we use the default [TensorFlow Distributed][tf-distributed] MirrorStrategy with the [NCCL][nccl]/[RCCL][rccl] all-reduce algorithm.

The benchmark task is training ResNet50-v1 on a synthetic ImageNet dataset using a momentum optimizer.
This compute-heavy task is characteristic of many other deep computer vision tasks with its dense image inputs and a deep, feed-forward, mostly convolutional architecture that translates well to GPGPUs.

[tf_cnn_benchmarks]: https://github.com/tensorflow/benchmarks
[nccl]: https://developer.nvidia.com/nccl
[rccl]: https://github.com/ROCmSoftwarePlatform/rccl
[tf-distributed]: https://www.tensorflow.org/guide/distributed_training

##### Singularity Containers

We used the HPC-oriented container platform [Singularity][singularity] (v3.5.2) to manage our environment and dependencies for this study.
Singularity ≥3.5 is required for ROCm support.

All reported results were collected using official TensorFlow and ROCm images available on Docker Hub. Singularity images can be pulled with:

```bash
$ singularity pull docker://$IMAGE_WITH_TAG
```

You can start a shell or run a script with:
```bash
# start a shell in the container environment w/ NVIDIA GPU access
$ singularity shell --nv $PATH_TO_SIMG

# run a python script in the container environment w/ ROCm GPU access
$ singularity exec --rocm $PATH_TO_SIMG python3 run.py
```

The following containers were used to collect reported results:

* **CUDA:** `tensorflow/tensorflow:1.15.2-gpu-py3` ([link][cuda-image])
* **ROCm:** `rocm/tensorflow:rocm3.1-tf1.15-dev` ([link][rocm-image])

[singularity]: https://sylabs.io/guides/3.5/user-guide/
[cuda-image]: https://hub.docker.com/layers/tensorflow/tensorflow/1.15.2-gpu-py3/images/sha256-da7b6c8a63bdafa77864e7e874664acfe939fdc140cb99940610c34b8c461cd0?context=explore
[rocm-image]: https://hub.docker.com/layers/rocm/tensorflow/rocm3.1-tf1.15-dev/images/sha256-21bad6c4225f92253fdcd5db4ba24ae850ee7b6e294cd23448ccd7288498f9b5?context=explore


##### Training Throughput

We measured computational performance of each system using **training images per second**.

An iteration includes both forward and backward passes through the network.
We used the largest power-of-2 batch size that would fit in GPU memory: 64 images/device for the GTX and RTX systems (11gb) and 256 images/device for the V100 and MI50 systems (32gb).
We ran enough warm-up iterations for the training speed to appear stable (5 steps for the NVIDIA hardware and 100 steps for AMD hardware).
The final training throughput is the median of three runs with 500 steps each.

The following script will run ResNet50 training benchmarks on 1-8 GPUs.
Fill out the variables at the top (`container_path`, `gpu_flag`, and `batch_size`) based on your specific system.

```bash
container_path=...
gpu_flag=...  # --nv or --rocm
batch_size=... # 64 for gtx or rtx (11gb), 256 for mi50 or v100 (32gb)

# run benchmarks on 1-8 GPUs
for n in {1..8}; do
    singularity exec $gpu_flag $container_path \
        python tf_cnn_benchmarks.py --num_gpus $n --batch_size $batch_size \
            --variable_update replicated --all_reduce_spec nccl \
            --model resnet50 --data_name imagenet --optimizer momentum \
            --nodistortions --gradient_repacking 1 --ml_perf
done
```

Some other helpful environment variables:
  * `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` control which GPUs are visible to TensorFlow.
  * `NCCL_DEBUG=INFO` will print out the GPU-GPU interconnects and NCCL ring topology used for the all-reduce operations, which is useful for verification purposes.
  * `NCCL_P2P_LEVEL` controls when to use direct GPU-to-GPU transport by setting the max allowable distance.
    A value of 0 (or LOC) disables all P2P communications.
  * `TF_XLA_FLAGS=--tf_xla_auto_jit=2` will force XLA compilation, optimizing the graph for your given hardware.
    This is particularly effective in mixed-precision mode when using GPUs with Tensor Cores.
  * [Other NCCL flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

Some other helpful benchmark options:
  * `--trace_file=trace.json` will save a tfprof trace of your training process, averaged over the first 10 steps.
    The results can be viewed at `chrome://tracing` in the Chrome browser.
    This is useful for debugging distributed performance issues.
  * `--use_fp16` will run the training in mixed-precision mode.
    This will use NVIDIA Tensor Cores on supported hardware.
  * Full benchmark options can be listed with `python tf_cnn_benchmarks.py --helpfull`.

##### Power Efficiency

Performance per Watt is an extremely important metric when evaluating HPC systems.
This is often reported in FLOPS/W (Floating Point Operations per Second per Watt) using a benchmark such as LINPACK.
For this study, we use a practical machine learning analog: **training images per second per Watt**.

We approximate power consumption as

$$\text{Avg. RAPL Power Draw (non-GPU)} + \sum^\text{all GPUs}\text{Avg. GPU Power Draw}$$

* **Non-GPU:** *Running Average Power Limit*, or *RAPL*, is an Intel processor feature that provides information on energy and power consumption of different physical domains.
  Average power draw was collected using the powercap interface: we queried `energy_uj` once per second over a 1-minute interval of a given workload, calculating average power over each timestep pair.
  Power data was collected over package-0 (core), package-1 (uncore), and the DRAM power plane.
  This excludes GPU power draw, which was recorded separately.

  We collected our data using code modified from [the `powerstat` tool](https://github.com/ColinIanKing/powerstat).
  Other utilities for accessing RAPL metrics include `perf`, `turbostat`, or `powertop`.

* **GPU:** Average GPU power draw was collected using the `nvidia-smi` and `rocm-smi` utilities.

  ```
  # For NVIDIA
  timeout 60 nvidia-smi --query-gpu=timestamp,name,index,power.draw --format=csv --loop=1 -f $LOGFILE

  # For ROCm
  for i in {1..60}; do rocm-smi -P --json >> $LOGFILE; sleep 1; done
  ```

  The power consumption measurements can be retrieved and averaged from these files.

The utility scripts that we used can be found [here](#).
To collect power information for different modes,
we started a training run as described in [Training Throughput](#training-throughput),
waited until iteration 10 of training,
then manually started the power consumption monitoring tools.


### Citation

This poster was presented at [ISC High Performance](https://www.isc-hpc.com/) in June 2020.
To cite our findings, please use:

```bibtex
@misc{ obenschain20,
       author = "Keith Obenschain and Douglas Schwer and Alisha Sharma",
       title = "Initial assessment of the AMD MI50 GPGPUs for scientific and machine learning applications",
       year = "2020",
       howpublished = "Research poster presented at ISC High Performance 2020" }
```


### Contact

For any questions, please contact the authors at [emerging_architectures@nrl.navy.mil](mailto:emerging_architectures@nrl.navy.mil).
