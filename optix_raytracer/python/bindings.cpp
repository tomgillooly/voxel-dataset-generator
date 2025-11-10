#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "voxel_tracer.h"

namespace py = pybind11;

/**
 * Python wrapper for VoxelRayTracer with numpy array support
 */
class PyVoxelRayTracer {
public:
    PyVoxelRayTracer(py::array_t<uint8_t> voxel_grid, float voxel_size = 1.0f) {
        // Get buffer info
        py::buffer_info buf = voxel_grid.request();

        if (buf.ndim != 3) {
            throw std::runtime_error("Voxel grid must be a 3D array");
        }

        // Extract dimensions (assuming Z, Y, X order from numpy)
        int res_z = static_cast<int>(buf.shape[0]);
        int res_y = static_cast<int>(buf.shape[1]);
        int res_x = static_cast<int>(buf.shape[2]);

        // Convert to vector (flatten in row-major order matching numpy)
        std::vector<unsigned char> voxel_data;
        voxel_data.reserve(res_x * res_y * res_z);

        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        for (int z = 0; z < res_z; z++) {
            for (int y = 0; y < res_y; y++) {
                for (int x = 0; x < res_x; x++) {
                    int idx = z * (res_y * res_x) + y * res_x + x;
                    voxel_data.push_back(ptr[idx]);
                }
            }
        }

        // Create C++ tracer
        m_tracer = std::make_unique<VoxelRayTracer>(
            voxel_data, res_x, res_y, res_z, voxel_size
        );
    }

    py::array_t<float> trace_rays(py::array_t<float> origins,
                                   py::array_t<float> directions) {
        // Validate input shapes
        py::buffer_info origins_buf = origins.request();
        py::buffer_info directions_buf = directions.request();

        if (origins_buf.ndim < 2 || origins_buf.shape[origins_buf.ndim - 1] != 3) {
            throw std::runtime_error("Origins must have shape (..., 3)");
        }

        if (directions_buf.ndim < 2 || directions_buf.shape[directions_buf.ndim - 1] != 3) {
            throw std::runtime_error("Directions must have shape (..., 3)");
        }

        // Compute total number of rays
        size_t num_rays_origins = 1;
        size_t num_rays_directions = 1;

        for (int i = 0; i < origins_buf.ndim - 1; i++) {
            num_rays_origins *= origins_buf.shape[i];
        }

        for (int i = 0; i < directions_buf.ndim - 1; i++) {
            num_rays_directions *= directions_buf.shape[i];
        }

        if (num_rays_origins != num_rays_directions) {
            throw std::runtime_error("Number of origins and directions must match");
        }

        int num_rays = static_cast<int>(num_rays_origins);

        // Convert to vectors
        std::vector<float> origins_vec(origins_buf.size);
        std::vector<float> directions_vec(directions_buf.size);

        float* origins_ptr = static_cast<float*>(origins_buf.ptr);
        float* directions_ptr = static_cast<float*>(directions_buf.ptr);

        std::copy(origins_ptr, origins_ptr + origins_buf.size, origins_vec.begin());
        std::copy(directions_ptr, directions_ptr + directions_buf.size, directions_vec.begin());

        // Trace rays
        std::vector<float> distances = m_tracer->traceRays(
            origins_vec, directions_vec, num_rays
        );

        // Convert back to numpy array with original shape (minus last dimension)
        std::vector<ssize_t> result_shape;
        for (int i = 0; i < origins_buf.ndim - 1; i++) {
            result_shape.push_back(origins_buf.shape[i]);
        }

        py::array_t<float> result(result_shape);
        py::buffer_info result_buf = result.request();
        float* result_ptr = static_cast<float*>(result_buf.ptr);

        std::copy(distances.begin(), distances.end(), result_ptr);

        return result;
    }

    void set_voxel_grid(py::array_t<uint8_t> voxel_grid) {
        py::buffer_info buf = voxel_grid.request();

        if (buf.ndim != 3) {
            throw std::runtime_error("Voxel grid must be a 3D array");
        }

        int res_z = static_cast<int>(buf.shape[0]);
        int res_y = static_cast<int>(buf.shape[1]);
        int res_x = static_cast<int>(buf.shape[2]);

        std::vector<unsigned char> voxel_data;
        voxel_data.reserve(res_x * res_y * res_z);

        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        for (int z = 0; z < res_z; z++) {
            for (int y = 0; y < res_y; y++) {
                for (int x = 0; x < res_x; x++) {
                    int idx = z * (res_y * res_x) + y * res_x + x;
                    voxel_data.push_back(ptr[idx]);
                }
            }
        }

        m_tracer->updateVoxelGrid(voxel_data, res_x, res_y, res_z);
    }

    py::dict get_grid_info() {
        int res_x, res_y, res_z;
        float voxel_size;
        m_tracer->getGridInfo(res_x, res_y, res_z, voxel_size);

        py::dict info;
        info["resolution"] = py::make_tuple(res_z, res_y, res_x);  // Return in numpy order
        info["voxel_size"] = voxel_size;

        return info;
    }

    bool is_ready() const {
        return m_tracer && m_tracer->isReady();
    }

private:
    std::unique_ptr<VoxelRayTracer> m_tracer;
};

PYBIND11_MODULE(optix_voxel_tracer, m) {
    m.doc() = "OptiX-based voxel ray tracer for transparent distance accumulation";

    py::class_<PyVoxelRayTracer>(m, "VoxelRayTracer")
        .def(py::init<py::array_t<uint8_t>, float>(),
             py::arg("voxel_grid"),
             py::arg("voxel_size") = 1.0f,
             R"pbdoc(
                Create a voxel ray tracer.

                Parameters
                ----------
                voxel_grid : numpy.ndarray
                    3D boolean or uint8 array representing voxel occupancy.
                    Shape: (Z, Y, X) in standard numpy ordering.
                voxel_size : float, optional
                    Physical size of each voxel (default: 1.0)

                Examples
                --------
                >>> import numpy as np
                >>> from optix_voxel_tracer import VoxelRayTracer
                >>> voxels = np.load("object_0001/level_0.npz")['voxels']
                >>> tracer = VoxelRayTracer(voxels)
             )pbdoc")

        .def("trace_rays", &PyVoxelRayTracer::trace_rays,
             py::arg("origins"),
             py::arg("directions"),
             R"pbdoc(
                Trace rays through the voxel grid.

                Parameters
                ----------
                origins : numpy.ndarray
                    Ray origins with shape (..., 3) where last dimension is (x, y, z)
                directions : numpy.ndarray
                    Ray directions with shape (..., 3) where last dimension is (x, y, z)
                    Directions should be normalized.

                Returns
                -------
                distances : numpy.ndarray
                    Accumulated distances through occupied voxels.
                    Shape matches input shape without last dimension.

                Examples
                --------
                >>> origins = np.zeros((512, 512, 3), dtype=np.float32)
                >>> directions = np.zeros((512, 512, 3), dtype=np.float32)
                >>> # ... fill origins and directions ...
                >>> distances = tracer.trace_rays(origins, directions)
                >>> print(distances.shape)  # (512, 512)
             )pbdoc")

        .def("set_voxel_grid", &PyVoxelRayTracer::set_voxel_grid,
             py::arg("voxel_grid"),
             "Update the voxel grid without recreating the tracer")

        .def("get_grid_info", &PyVoxelRayTracer::get_grid_info,
             "Get information about the current voxel grid")

        .def("is_ready", &PyVoxelRayTracer::is_ready,
             "Check if the tracer is ready to use");

    // Module-level documentation
    m.attr("__version__") = "0.1.0";
}
