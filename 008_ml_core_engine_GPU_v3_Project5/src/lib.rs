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
            
            let info = adapter.get_info();
            println!("🚀 ML Core Engine initialized successfully on GPU: {:?} (Backend: {:?})", info.name, info.backend);

            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
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
        // رفض الأبعاد الصفرية برمجياً لتجاوز تنبيهات الـ invalid_inputs الصامتة
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
        if self.rows == 0 || self.cols == 0 || other.cols == 0 {
            return Err(PyValueError::new_err("GPU matmul does not support zero-sized dimensions"));
        }

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let result_len = m * n;
        let self_data = self.data.clone();
        let other_data = other.data.clone();

        let ctx = get_gpu_context();
        let device = &ctx.device;
        let queue = &ctx.queue;

        let result_data = pollster::block_on(async {
            let uniforms = MatMulUniforms { m: m as u32, k: k as u32, n: n as u32, use_relu: if use_relu { 1 } else { 0 } };
            
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniforms"), contents: bytemuck::bytes_of(&uniforms), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A"), contents: bytemuck::cast_slice(&self_data), usage: wgpu::BufferUsages::STORAGE,
            });
            let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B"), contents: bytemuck::cast_slice(&other_data), usage: wgpu::BufferUsages::STORAGE,
            });
            let buffer_result = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Res"), size: (result_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
            });
            let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Stage"), size: (result_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fused MatMul + ReLU Shader"),
                source: wgpu::ShaderSource::Wgsl(r#"
                    struct Uniforms { m: u32, k: u32, n: u32, use_relu: u32 };
                    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                    @group(0) @binding(1) var<storage, read> mat_a: array<f32>;
                    @group(0) @binding(2) var<storage, read> mat_b: array<f32>;
                    @group(0) @binding(3) var<storage, read_write> mat_c: array<f32>;

                    @compute @workgroup_size(16, 16)
                    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                        let row = id.x;
                        let col = id.y;
                        if (row >= uniforms.m || col >= uniforms.n) { return; }
                        
                        var sum = 0.0;
                        for (var i = 0u; i < uniforms.k; i = i + 1u) {
                            sum += mat_a[row * uniforms.k + i] * mat_b[i * uniforms.n + col];
                        }
                        if (uniforms.use_relu == 1u) { sum = max(0.0, sum); }
                        mat_c[row * uniforms.n + col] = sum;
                    }
                "#.into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None, layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] })),
                module: &shader, entry_point: "main",
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buffer_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buffer_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buffer_result.as_entire_binding() },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((m as u32 + 15) / 16, (n as u32 + 15) / 16, 1);
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

            Ok::<Vec<f32>, pyo3::PyErr>(result)
        })?;

        Ok(PyMatrix { rows: m, cols: n, data: result_data })
    }

    // نواة GPU مخصصة لحساب مشتقة الـ ReLU مباشرة وتجنب التنقل بين CPU و GPU
    fn gpu_relu_backward(&self, output_vals: &PyMatrix) -> PyResult<PyMatrix> {
        if self.rows != output_vals.rows || self.cols != output_vals.cols {
            return Err(PyValueError::new_err("Dimensions mismatch for gpu_relu_backward"));
        }
        let rows = self.rows;
        let cols = self.cols;
        let size = rows * cols;
        let self_data = self.data.clone();
        let out_data = output_vals.data.clone();

        let ctx = get_gpu_context();
        let device = &ctx.device;
        let queue = &ctx.queue;

        let result_data = pollster::block_on(async {
            let uniforms = ReluBwUniforms { size: size as u32 };
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RBw Uniforms"), contents: bytemuck::bytes_of(&uniforms), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let buffer_dvals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DValues"), contents: bytemuck::cast_slice(&self_data), usage: wgpu::BufferUsages::STORAGE,
            });
            let buffer_outs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Outs"), contents: bytemuck::cast_slice(&out_data), usage: wgpu::BufferUsages::STORAGE,
            });
            let buffer_result = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Res"), size: (size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
            });
            let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Stage"), size: (size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ReLU Backward Shader"),
                source: wgpu::ShaderSource::Wgsl(r#"
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
                "#.into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None, layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] })),
                module: &shader, entry_point: "main",
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buffer_dvals.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buffer_outs.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buffer_result.as_entire_binding() },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((size as u32 + 255) / 256, 1, 1);
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

            Ok::<Vec<f32>, pyo3::PyErr>(result)
        })?;

        Ok(PyMatrix { rows, cols, data: result_data })
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