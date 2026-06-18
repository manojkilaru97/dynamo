// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python exception types mirroring Dynamo's [`ErrorType`] enum.
//!
//! The [`define_dynamo_exceptions!`] macro auto-generates a Python exception class
//! for each Dynamo error variant, a conversion function from Python exceptions back
//! to [`DynamoError`], and a registration function for the `_core` module.
//!
//! When new variants are added to [`ErrorType`] or [`BackendError`], add a
//! corresponding entry to the macro invocation below to keep Python exceptions
//! in sync.

use dynamo_runtime::error::{BackendError, ErrorType};
use pyo3::prelude::*;
use pyo3::types::PyModule;

// Base exception for all Dynamo errors.
pyo3::create_exception!(dynamo._core, DynamoException, pyo3::exceptions::PyException);

/// Defines Python exception classes for each Dynamo error type.
///
/// For each `(RustExceptionName, BackendError)` pair, the macro:
/// 1. Creates a Python exception class inheriting from `DynamoException`
/// 2. Adds it to `py_exception_to_backend_error()` for Python → `ErrorType` extraction
/// 3. Adds it to `register_exceptions()` for module registration
///
/// The conversion intentionally returns a `BackendError` variant and message
/// rather than a fully constructed `DynamoError`. This lets the caller decide
/// how to wrap it — backend contexts use `ErrorType::Backend(...)`, while
/// other contexts could map to top-level `ErrorType` variants.
macro_rules! define_dynamo_exceptions {
    ( $( ($name:ident, $backend_error:expr) ),* $(,)? ) => {
        $(
            pyo3::create_exception!(dynamo._core, $name, DynamoException);
        )*

        /// Extract an [`ErrorType`] variant from a Python exception if it is
        /// a known Dynamo exception.
        ///
        /// Returns `Some((ErrorType, message))` if the exception is a Dynamo
        /// exception, `None` otherwise.
        pub fn py_exception_to_backend_error(
            py: Python<'_>,
            err: &PyErr,
        ) -> Option<(ErrorType, String)> {
            // Check specific subtypes first (most-specific match wins).
            $(
                if err.is_instance_of::<$name>(py) {
                    let message = err
                        .value(py)
                        .str()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    return Some(($backend_error, message));
                }
            )*

            // Fall back: check if it's a bare DynamoException (Unknown).
            if err.is_instance_of::<DynamoException>(py) {
                let message = err
                    .value(py)
                    .str()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Some((ErrorType::Backend(BackendError::Unknown), message));
            }

            None
        }

        /// Register all Dynamo exception classes on the `_core` module.
        pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
            m.add("DynamoException", m.py().get_type::<DynamoException>())?;
            $(
                m.add(stringify!($name), m.py().get_type::<$name>())?;
            )*
            Ok(())
        }
    };
}

// ---------------------------------------------------------------------------
// Exception definitions — one entry per ErrorType variant exposed to Python.
//
// All error types are exposed to Python as exception classes. When raised by
// Python backend code, they are interpreted as Backend* errors in Rust
// (e.g., raising `InvalidArgument` in Python becomes `BackendInvalidArgument`
// on the Rust side).
//
// When a new variant is added to BackendError in error.rs, add a
// corresponding line here so that a Python exception is generated.
// ---------------------------------------------------------------------------
define_dynamo_exceptions!(
    (Unknown, ErrorType::Backend(BackendError::Unknown)),
    (
        InvalidArgument,
        ErrorType::Backend(BackendError::InvalidArgument)
    ),
    (
        CannotConnect,
        ErrorType::Backend(BackendError::CannotConnect)
    ),
    (Disconnected, ErrorType::Backend(BackendError::Disconnected)),
    (
        ConnectionTimeout,
        ErrorType::Backend(BackendError::ConnectionTimeout)
    ),
    (Cancelled, ErrorType::Backend(BackendError::Cancelled)),
    (
        EngineShutdown,
        ErrorType::Backend(BackendError::EngineShutdown)
    ),
    (
        StreamIncomplete,
        ErrorType::Backend(BackendError::StreamIncomplete)
    ),
    (ResourceExhausted, ErrorType::ResourceExhausted),
);
