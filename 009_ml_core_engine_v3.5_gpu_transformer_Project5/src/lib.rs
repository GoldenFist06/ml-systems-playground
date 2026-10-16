use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pollster;
use wgpu::util::DeviceExt;
use std::sync::OnceLock;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

fn get_gpu_context() -> &'static GpuContext {
    GPU_CONTEXT.get_or_init(|| {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }).await.expect("Failed to find a suitable GPU adapter");
            
            let limits = adapter.limits();
            
            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    label: Some("ML Core GPU Device"),
                },
                None,
            ).await.expect("Failed to create GPU device and queue");
            
            GpuContext { device, queue }
        })
    })
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulUniforms {
    m: u32,
    k: u32,
    n: u32,
    use_relu: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReluBwUniforms {
    size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsNormUniforms {
    rows: u32,
    cols: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SwiGluUniforms {
    size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxUniforms {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeUniforms {
    rows: u32,
    cols: u32,
}

fn run_compute_shader(
    label: &str,
    wgsl_source: &str,
    uniform_bytes: &[u8],
    storage_inputs: &[&[f32]],
    output_size: usize,
    workgroups: (u32, u32, u32),
) -> Vec<f32> {
    let ctx = get_gpu_context();
    let device = &ctx.device;
    let queue = &ctx.queue;

    pollster::block_on(async {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut buffers = Vec::new();
        for (idx, input_data) in storage_inputs.iter().enumerate() {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Input {}", idx)),
                contents: bytemuck::cast_slice(input_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            buffers.push(buf);
        }

        let mut bgl_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ];

        let mut bg_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }
        ];

        for (idx, buf) in buffers.iter().enumerate() {
            bgl_entries.push(wgpu::BindGroupLayoutEntry {
                binding: (idx + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            bg_entries.push(wgpu::BindGroupEntry {
                binding: (idx + 1) as u32,
                resource: buf.as_entire_binding(),
            });
        }

        let result_binding_idx = (storage_inputs.len() + 1) as u32;
        let buffer_result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        bgl_entries.push(wgpu::BindGroupLayoutEntry {
            binding: result_binding_idx,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        bg_entries.push(wgpu::BindGroupEntry {
            binding: result_binding_idx,
            resource: buffer_result.as_entire_binding(),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &bg_entries,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        encoder.copy_buffer_to_buffer(&buffer_result, 0, &buffer_staging, 0, buffer_result.size());
        queue.submit(Some(encoder.finish()));

        let slice = buffer_staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| { sender.send(v).unwrap(); });
        device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer_staging.unmap();

        result
    })
}

#[pymethods]
impl PyMatrix {
    #[new]
    fn new(rows: usize, cols: usize, initial_value: f32) -> PyResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err("Matrix dimensions cannot be zero"));
        }
        let data = vec![initial_value; rows * cols];
        Ok(PyMatrix { rows, cols, data })
    }

    #[staticmethod]
    fn from_list(rows: usize, cols: usize, data: Vec<f32>) -> PyResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err("Matrix dimensions cannot be zero"));
        }
        if data.len() != rows * cols {
            return Err(PyValueError::new_err(format!(
                "Data length ({}) does not match dimensions {}x{}={}",
                data.len(), rows, cols, rows * cols
            )));
        }
        Ok(PyMatrix { rows, cols, data })
    }

    fn __repr__(&self) -> String {
        format!("PyMatrix(rows={}, cols={}, shape=({}, {}))", self.rows, self.cols, self.rows, self.cols)
    }

    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }

    #[getter]
    fn cols(&self) -> usize {
        self.cols
    }

    fn add(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyValueError::new_err("Matrix dimensions must match for addition"));
        }
        let result_data: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Ok(PyMatrix { rows: self.rows, cols: self.cols, data: result_data })
    }

    fn sub(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyValueError::new_err("Matrix dimensions must match for subtraction"));
        }
        let result_data: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Ok(PyMatrix { rows: self.rows, cols: self.cols, data: result_data })
    }

    fn multiply(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyValueError::new_err("Matrix dimensions must match for element-wise multiplication"));
        }
        let result_data: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Ok(PyMatrix { rows: self.rows, cols: self.cols, data: result_data })
    }

    fn scale(&self, factor: f32) -> PyMatrix {
        let result_data: Vec<f32> = self.data.iter().map(|&val| val * factor).collect();
        PyMatrix { rows: self.rows, cols: self.cols, data: result_data }
    }

    fn matmul(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.cols != other.rows {
            return Err(PyValueError::new_err(format!("Cannot multiply matrices of shape ({}, {}) and ({}, {})", self.rows, self.cols, other.rows, other.cols)));
        }
        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;
        let mut result_data = vec![0.0; rows * cols];
        for i in 0..rows {
            for k in 0..inner {
                let a_val = self.data[i * inner + k];
                for j in 0..cols {
                    result_data[i * cols + j] += a_val * other.data[k * cols + j];
                }
            }
        }
        Ok(PyMatrix { rows, cols, data: result_data })
    }

    #[pyo3(signature = (other, use_relu = false))]
    fn gpu_matmul(&self, other: &PyMatrix, use_relu: bool) -> PyResult<PyMatrix> {
        if self.cols != other.rows {
            return Err(PyValueError::new_err("Dimensions mismatch for GPU matmul"));
        }
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let uniforms = MatMulUniforms { m: m as u32, k: k as u32, n: n as u32, use_relu: if use_relu { 1 } else { 0 } };
        let wgsl = r#"
            struct Uniforms { m: u32, k: u32, n: u32, use_relu: u32 };
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> mat_a: array<f32>;
            @group(0) @binding(2) var<storage, read> mat_b: array<f32>;
            @group(0) @binding(3) var<storage, read_write> mat_c: array<f32>;

            const TILE_SIZE: u32 = 16u;
            var<workgroup> tile_a: array<array<f32, 16>, 16>;
            var<workgroup> tile_b: array<array<f32, 16>, 16>;

            @compute @workgroup_size(16, 16)
            fn main(
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(global_invocation_id) global_id: vec3<u32>
            ) {
                let row = global_id.x;
                let col = global_id.y;
                let local_row = local_id.x;
                let local_col = local_id.y;

                var sum = 0.0;
                let num_tiles = (uniforms.k + TILE_SIZE - 1u) / TILE_SIZE;

                for (var t = 0u; t < num_tiles; t = t + 1u) {
                    let tiled_col = t * TILE_SIZE + local_col;
                    let tiled_row = t * TILE_SIZE + local_row;

                    if (row < uniforms.m && tiled_col < uniforms.k) {
                        tile_a[local_row][local_col] = mat_a[row * uniforms.k + tiled_col];
                    } else {
                        tile_a[local_row][local_col] = 0.0;
                    }

                    if (tiled_row < uniforms.k && col < uniforms.n) {
                        tile_b[local_row][local_col] = mat_b[tiled_row * uniforms.n + col];
                    } else {
                        tile_b[local_row][local_col] = 0.0;
                    }

                    workgroupBarrier();

                    for (var i = 0u; i < TILE_SIZE; i = i + 1u) {
                        sum += tile_a[local_row][i] * tile_b[i][local_col];
                    }

                    workgroupBarrier();
                }

                if (row < uniforms.m && col < uniforms.n) {
                    var final_val = sum;
                    if (uniforms.use_relu == 1u) {
                        final_val = max(0.0, final_val);
                    }
                    mat_c[row * uniforms.n + col] = final_val;
                }
            }
        "#;

        let result_data = run_compute_shader(
            "MatMul Shader",
            wgsl,
            bytemuck::bytes_of(&uniforms),
            &[&self.data, &other.data],
            m * n,
            ((m as u32 + 15) / 16, (n as u32 + 15) / 16, 1),
        );

        Ok(PyMatrix { rows: m, cols: n, data: result_data })
    }

    fn gpu_relu_backward(&self, output_vals: &PyMatrix) -> PyResult<PyMatrix> {
        if self.rows != output_vals.rows || self.cols != output_vals.cols {
            return Err(PyValueError::new_err("Dimensions mismatch for gpu_relu_backward"));
        }
        let size = self.rows * self.cols;
        let uniforms = ReluBwUniforms { size: size as u32 };
        let wgsl = r#"
            struct Uniforms { size: u32 };
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> dvalues: array<f32>;
            @group(0) @binding(2) var<storage, read> outs: array<f32>;
            @group(0) @binding(3) var<storage, read_write> result: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if (idx >= uniforms.size) { return; }
                var mask = 0.0;
                if (outs[idx] > 0.0) { mask = 1.0; }
                result[idx] = dvalues[idx] * mask;
            }
        "#;

        let result_data = run_compute_shader(
            "ReLU Backward Shader",
            wgsl,
            bytemuck::bytes_of(&uniforms),
            &[&self.data, &output_vals.data],
            size,
            ((size as u32 + 255) / 256, 1, 1),
        );

        Ok(PyMatrix { rows: self.rows, cols: self.cols, data: result_data })
    }

    #[pyo3(signature = (eps = 1e-5))]
    fn gpu_rms_norm(&self, eps: f32) -> PyResult<PyMatrix> {
        let size = self.rows * self.cols;
        let uniforms = RmsNormUniforms { rows: self.rows as u32, cols: self.cols as u32, eps };
        let wgsl = r#"
            struct Uniforms { rows: u32, cols: u32, eps: f32 };
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> input_data: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

            var<workgroup> shared_sum: array<f32, 256>;

            @compute @workgroup_size(256)
            fn main(
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) wg_id: vec3<u32>
            ) {
                let row = wg_id.x;
                if (row >= uniforms.rows) { return; }
                let tid = local_id.x;
                let cols = uniforms.cols;

                var sum_sq = 0.0;
                var i = tid;
                while (i < cols) {
                    let val = input_data[row * cols + i];
                    sum_sq += val * val;
                    i += 256u;
                }

                shared_sum[tid] = sum_sq;
                workgroupBarrier();

                for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_sum[tid] += shared_sum[tid + stride];
                    }
                    workgroupBarrier();
                }

                var rms = 0.0;
                if (tid == 0u) {
                    let mean_sq = shared_sum[0u] / f32(cols);
                    rms = inverseSqrt(mean_sq + uniforms.eps);
                    shared_sum[0u] = rms;
                }
                workgroupBarrier();

                rms = shared_sum[0u];
                i = tid;
                while (i < cols) {
                    let idx = row * cols + i;
                    output_data[idx] = input_data[idx] * rms;
                    i += 256u;
                }
            }
        "#;

        let result_data = run_compute_shader(
            "RMSNorm Shader",
            wgsl,
            bytemuck::bytes_of(&uniforms),
            &[&self.data],
            size,
            (self.rows as u32, 1, 1),
        );

        Ok(PyMatrix { rows: self.rows, cols: self.cols, data: result_data })
    }

    fn gpu_swiglu(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        let size = self.rows * self.cols;
        let uniforms = SwiGluUniforms { size: size as u32 };
        let wgsl = r#"
            struct Uniforms { size: u32 };
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> gate_data: array<f32>;
            @group(0) @binding(2) var<storage, read> up_data: array<f32>;
            @group(0) @binding(3) var<storage, read_write> output_data: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if (idx >= uniforms.size) { return; }
                let x = gate_data[idx];
                let y = up_data[idx];
                let silu = x / (1.0 + exp(-x));
                output_data[idx] = silu * y;
            }
        "#;

        let result_data = run_compute_shader(
            "SwiGLU Shader",
            wgsl,
            bytemuck::bytes_of(&uniforms),
            &[&self.data, &other.data],
            size,
            ((size as u32 + 255) / 256, 1, 1),
        );

        Ok(PyMatrix { rows: self.rows, cols: self.cols, data: result_data })
    }

    fn gpu_softmax(&self) -> PyResult<PyMatrix> {
        let size = self.rows * self.cols;
        let uniforms = SoftmaxUniforms { rows: self.rows as u32, cols: self.cols as u32 };
        let wgsl = r#"
            struct Uniforms { rows: u32, cols: u32 };
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> input_data: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

            var<workgroup> shared_val: array<f32, 256>;

            @compute @workgroup_size(256)
            fn main(
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) wg_id: vec3<u32>
            ) {
                let row = wg_id.x;
                if (row >= uniforms.rows) { return; }
                let tid = local_id.x;
                let cols = uniforms.cols;

                var max_val = -3.402823466e+38;
                var i = tid;
                while (i < cols) {
                    max_val = max(max_val, input_data[row * cols + i]);
                    i += 256u;
                }
                shared_val[tid] = max_val;
                workgroupBarrier();

                for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_val[tid] = max(shared_val[tid], shared_val[tid + stride]);
                    }
                    workgroupBarrier();
                }
                let row_max = shared_val[0u];
                workgroupBarrier();

                var sum_exp = 0.0;
                i = tid;
                while (i < cols) {
                    let exp_val = exp(input_data[row * cols + i] - row_max);
                    sum_exp += exp_val;
                    output_data[row * cols + i] = exp_val;
                    i += 256u;
                }
                shared_val[tid] = sum_exp;
                workgroupBarrier();

                for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_val[tid] += shared_val[tid + stride];
                    }
                    workgroupBarrier();
                }
                let row_sum = shared_val[0u];
                workgroupBarrier();

                i = tid;
                while (i < cols) {
                    output_data[row * cols + i] = output_data[row * cols + i] / row_sum;
                    i += 256u;
                }
            }
        "#;

        let result_data = run_compute_shader(
            "Softmax Shader",
            wgsl,
            bytemuck::bytes_of(&uniforms),
            &[&self.data],
            size,
            (self.rows as u32, 1, 1),
        );

        Ok(PyMatrix { rows: self.rows, cols: self.cols, data: result_data })
    }

    fn gpu_rope(&self) -> PyResult<PyMatrix> {
        let size = self.rows * self.cols;
        let uniforms = RopeUniforms { rows: self.rows as u32, cols: self.cols as u32 };
        let wgsl = r#"
            struct Uniforms { rows: u32, cols: u32 };
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> input_data: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                let half_cols = uniforms.cols / 2u;
                let total_pairs = uniforms.rows * half_cols;
                if (idx >= total_pairs) { return; }

                let row = idx / half_cols;
                let i = idx % half_cols;

                let x_idx = row * uniforms.cols + (2u * i);
                let y_idx = x_idx + 1u;

                let x = input_data[x_idx];
                let y = input_data[y_idx];

                let freq = 1.0 / pow(10000.0, f32(2u * i) / f32(uniforms.cols));
                let m = f32(row);
                let val = m * freq;
                let cos_val = cos(val);
                let sin_val = sin(val);

                output_data[x_idx] = x * cos_val - y * sin_val;
                output_data[y_idx] = x * sin_val + y * cos_val;
            }
        "#;

        let total_pairs = (self.rows * (self.cols / 2)) as u32;
        let result_data = run_compute_shader(
            "RoPE Shader",
            wgsl,
            bytemuck::bytes_of(&uniforms),
            &[&self.data],
            size,
            ((total_pairs + 255) / 256, 1, 1),
        );

        Ok(PyMatrix { rows: self.rows, cols: self.cols, data: result_data })
    }

    fn gpu_attention(&self, key: &PyMatrix, value: &PyMatrix) -> PyResult<PyMatrix> {
        if self.cols != key.cols {
            return Err(PyValueError::new_err("Q and K dimensions (cols) must match for attention"));
        }
        if key.rows != value.rows {
            return Err(PyValueError::new_err("K and V rows must match for attention"));
        }
        let d_k = self.cols as f32;
        let scale = 1.0 / d_k.sqrt();
        
        let k_t = key.transpose();
        let q_kt = self.gpu_matmul(&k_t, false)?;
        let scaled = q_kt.scale(scale);
        let weights = scaled.gpu_softmax()?;
        weights.gpu_matmul(value, false)
    }

    fn transpose(&self) -> PyMatrix {
        let new_rows = self.cols;
        let new_cols = self.rows;
        let mut transposed_data = vec![0.0; self.data.len()];
        for r in 0..self.rows {
            for c in 0..self.cols {
                transposed_data[c * new_cols + r] = self.data[r * self.cols + c];
            }
        }
        PyMatrix { rows: new_rows, cols: new_cols, data: transposed_data }
    }

    fn to_list(&self) -> Vec<f32> {
        self.data.clone()
    }
}

#[pyfunction]
fn check_gpu_device() -> PyResult<String> {
    let _ctx = get_gpu_context();
    Ok(String::from("🚀 GPU Device Active and Initialized via Persistent Context!"))
}

#[pymodule]
fn ml_core_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatrix>()?;
    m.add_function(wrap_pyfunction!(check_gpu_device, m)?)?;
    Ok(())
}