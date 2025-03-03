use clap::Parser;
use itertools::multizip;
use ndarray::{Array2, Axis};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3_ffi::c_str;
use std::collections::HashMap;
use std::f32::consts::E;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    prompt: String,

    #[arg(short, long, default_value_t = 5)]
    n_tokens_to_generate: usize,
}

#[derive(Debug)]
struct Params {
    encoder: Py<PyAny>,
    hparams: HashMap<String, Value>,
    params: HashMap<String, Value>,
    input_ids: Vec<i64>,
}

#[derive(Debug)]
enum Value {
    Int(i64),
    Hashmap(HashMap<String, Value>),
    List(Vec<Value>),
    NumpyArray(Array2<f32>),
}

/// Extracts a HashMap from a PyAny object, recursively handling nested 
/// dictionaries, lists, and NumPy arrays.
/// 
/// This function is used to convert Python data structures into a 
/// Rust-friendly representation, where the `Value` enum is used to represent
/// the various data types.
fn extract_hashmap(py_any: &pyo3::Bound<'_, PyAny>) -> PyResult<HashMap<String, Value>> {
    let dict = py_any.downcast::<PyDict>()?;

    let mut result = HashMap::new();

    for (key, value) in dict.iter() {
        let key_str = key.extract::<String>()?;
        let val = if value.is_instance_of::<PyDict>() {
            Value::Hashmap(extract_hashmap(&value)?)
        } else if value.is_instance_of::<PyList>() {
            let list = value.downcast::<PyList>()?;
            let mut vec = Vec::new();
            for item in list.iter() {
                if let Ok(dict) = item.downcast::<PyDict>() {
                    vec.push(Value::Hashmap(extract_hashmap(&dict)?));
                } else if let Ok(i) = item.extract::<i64>() {
                    vec.push(Value::Int(i));
                }
            }
            Value::List(vec)
        } else if let Ok(py_arr) = value.extract::<Py<PyArrayDyn<f32>>>() {
            let array = Python::with_gil(|py| {
                let array_ref = py_arr.bind(py);
                let mut res = array_ref.readonly().as_array().to_owned().into_dyn();
                if res.ndim() != 2 {
                    res = res.insert_axis(Axis(1));
                }
                res.into_dimensionality::<ndarray::Dim<[usize; 2]>>()
                    .unwrap()
            });
            Value::NumpyArray(array)
        } else if let Ok(i) = value.extract::<i64>() {
            Value::Int(i)
        } else {
            continue;
        };
        result.insert(key_str, val);
    }
    Ok(result)
}

/// Applies layer normalization to the input tensor `x` using the provided
/// scale `g` and bias `b` parameters.
///
/// The layer normalization is performed independently for each row (token)
/// of the input tensor `x`. The mean and variance are computed along the
/// feature dimension (axis 1 for each row) and then used to normalize each
/// row.
///
/// # Arguments
/// - `x`: The input tensor of shape `(n_seq, n_embd)` (n_seq x 768 for gpt2).
/// - `g`: The scale parameter of shape `(n_embd,)`.
/// - `b`: The bias parameter of shape `(n_embd,)`.
/// - `eps`: A small constant added to the variance to ensure numerical stability.
///
/// # Returns
/// The normalized tensor of shape `(n_seq, n_embd)`.
fn layer_norm(x: Array2<f32>, g: &Value, b: &Value, eps: f32) -> Result<Array2<f32>, String> {
    let g = match g {
        Value::NumpyArray(arr) => arr,
        other => Err(format!("Expected Hashmap for g, got {:?}", other))?,
    };
    let b = match b {
        Value::NumpyArray(arr) => arr,
        other => Err(format!("Expected Hashmap for b, got {:?}", other))?,
    };

    // mean is n_seq. Have to insert axis to add back second dimension to make it n_seq x 1
    // which allows proper broadcasting when we substract mean and divide by var + eps.
    // each row (i.e. each token) gets normalized independently
    let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let var = x.var_axis(Axis(1), 0.).insert_axis(Axis(1)); // n_seq x 1
    let x_norm = (x - mean) / (var + eps).sqrt();
    Ok(g.flatten().to_owned() * x_norm + b.flatten())
}


/// Applies a linear transformation to the input tensor `x` using the provided
/// weight `w` and bias `b` parameters.
///
/// # Arguments
/// - `x`: The input tensor of shape `(n_seq, n_embd)`.
/// - `w`: The weight tensor of shape `(n_embd, n_out)`.
/// - `b`: The bias tensor of shape `(n_out,)`.
///
/// # Returns
/// The transformed tensor of shape `(n_seq, n_out)`.
fn linear(x: Array2<f32>, w: &Value, b: &Value) -> Result<Array2<f32>, String> {
    let w = match w {
        Value::NumpyArray(arr) => arr,
        other => Err(format!("Expected NumpyArray for w, got {:?}", other))?,
    };
    let b = match b {
        Value::NumpyArray(arr) => arr,
        other => Err(format!("Expected NumpyArray for b, got {:?}", other))?,
    };
    Ok(x.dot(w) + b.flatten())
}


/// Applies the softmax function to the input 2D tensor `array`.
/// 
/// The softmax function is applied to each row of the input tensor independently.
/// The softmax function is defined as:
/// 
/// `softmax(x_i) = exp(x_i) / sum(exp(x_j))`
/// 
/// where `x_i` is the i-th element of the input tensor.
///
/// # Arguments
/// - `array`: The input 2D tensor of shape `(n_seq, n_embd)`.
///
/// # Returns
/// The transformed 2D tensor of the same shape as the input, with each row
/// containing the softmax of the corresponding row in the input.
pub fn softmax(mut array: Array2<f32>) -> Array2<f32> {
    for mut row in array.axis_iter_mut(Axis(0)) {
        for value in &mut row {
            *value = E.powf(*value);
        }

        let sum: f32 = row.iter().sum();

        for value in row {
            *value /= sum;
        }
    }
    array
}

/// Applies the Gaussian Error Linear Unit (GELU) activation function to the
/// input tensor `x`.
///
/// The GELU activation function is defined as:
///
/// `gelu(x) = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// It applies the GELU activation function to each element of the input
/// tensor `x`.
///
/// # Arguments
/// - `x`: The input tensor of shape `(n_seq, n_embd)`.
///
/// # Returns
/// The transformed tensor of the same shape as the input, with the GELU
/// activation function applied to each element.
fn gelu(x: Array2<f32>) -> Array2<f32> {
    let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    // use mapv to apply the function to each element of the array
    x.mapv(|x| 0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh()))
}

/// Applies a feed-forward neural network (FFN) to the input tensor `x`.
///
/// The FFN consists of two linear transformations with a GELU activation
/// function in between.
///
/// # Arguments
/// - `x`: The input tensor of shape `(n_seq, n_embd)`.
/// - `c_fc`: A hashmap containing the weights and biases for the first linear
/// transformation. Shape is `(n_embd, 4 * n_embd)`.
/// - `c_proj`: A hashmap containing the weights and biases for the second 
/// linear transformation. Shape is `(4 * n_embd, n_embd)`.
///
/// # Returns
/// The transformed tensor of the same shape as the input, after applying the FFN.
fn ffn(x: Array2<f32>, c_fc: &Value, c_proj: &Value) -> Result<Array2<f32>, String> {
    let c_fc = match c_fc {
        Value::Hashmap(hash) => hash,
        other => Err(format!("Expected Hashmap for c_fc, got {:?}", other))?,
    };
    let c_proj = match c_proj {
        Value::Hashmap(arr) => arr,
        other => Err(format!("Expected Hashmap for f_proj, got {:?}", other))?,
    };

    let ffn1 = gelu(linear(x.clone(), &c_fc["w"], &c_fc["b"])?);
    let ffn2 = linear(ffn1, &c_proj["w"], &c_proj["b"])?;
    Ok(ffn2)
}

/// Computes the attention mechanism over the input tensors `q`, `k`, and `v`,
/// using the provided `mask`.
///
/// # Arguments
/// - `q`: The query tensor of shape `(n_seq, n_embd)`.
/// - `k`: The key tensor of shape `(n_seq, n_embd)`.
/// - `v`: The value tensor of shape `(n_seq, n_embd)`.
/// - `mask`: The attention mask tensor of shape `(n_seq, n_seq)`.
///
/// # Returns
/// The transformed tensor of shape `(n_seq, n_embd)`, after applying the
/// attention mechanism.
fn attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, mask: &Array2<f32>) -> Array2<f32> {
    softmax((q.dot(&k.t())) / (q.dim().1 as f32).sqrt() + mask).dot(v)
}

/// Applies the multi-head attention (MHA) mechanism to the input tensor `x`.
///
/// The MHA mechanism first linearly transforms the input `x` using the weights
/// and biases specified in `c_attn`. It then splits the transformed tensor
/// into `n_head` parts along the last axis, computes the attention mechanism
/// on each part, and concatenates the results.
///
/// # Arguments
/// - `x`: The input tensor of shape `(n_seq, n_embd)`.
/// - `c_attn`: A hashmap containing the weights and biases for the linear 
/// transformation applied to `x`. q, k, and v are combined in c_attn. shape
/// is `(n_embd, 3 * n_embd)`
/// - `c_proj`: A hashmap containing the weights and biases for the final 
/// linear transformation. shape is `(n_embd, n_embd)`
/// - `n_head`: The number of attention heads to use.
///
/// # Returns
/// The transformed tensor of the same shape as the input, after applying the 
/// MHA mechanism.
fn mha(x: Array2<f32>, c_attn: &Value, c_proj: &Value, n_head: i64) -> Result<Array2<f32>, String> {
    let c_attn = match c_attn {
        Value::Hashmap(hash) => hash,
        other => Err(format!("Expected Hashmap for c_attn, got {:?}", other))?,
    };
    let c_proj = match c_proj {
        Value::Hashmap(hash) => hash,
        other => Err(format!("Expected Hashmap for c_proj, got {:?}", other))?,
    };

    let x = linear(x, &c_attn["w"], &c_attn["b"]).unwrap();
    // First split x into 3 parts along the last axis (q, k, v)
    let qkv = x.axis_chunks_iter(Axis(1), x.shape()[1] / 3);
    // Then split each of those parts into n_head pieces
    let qkv_heads: Vec<Vec<Array2<f32>>> = qkv
        .map(|chunk| {
            chunk
                .axis_chunks_iter(Axis(1), chunk.shape()[1] / n_head as usize)
                .map(|view| view.to_owned())
                .collect()
        })
        .collect();

    // Create causal mask to hide future tokens from being attended to.
    // Causal mask is a square matrix of shape (seq_len, seq_len) where the 
    // left diagonal side is 1 and the rest is -1e10.
    let seq_len = x.shape()[0];
    let mut causal_mask = Array2::<f32>::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            causal_mask[[i, j]] = -1e10;
        }
    }

    let mut out_heads: Vec<Array2<f32>> = Vec::new();
    for (q, k, v) in multizip((
        qkv_heads[0].iter(),
        qkv_heads[1].iter(),
        qkv_heads[2].iter(),
    )) {
        out_heads.push(attention(q, k, v, &causal_mask));
    }

    // merge heads
    let merged_out_heads = ndarray::concatenate(
        Axis(1),
        &out_heads.iter().map(|x| x.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    // out projection
    let x = linear(merged_out_heads, &c_proj["w"], &c_proj["b"]).unwrap();
    Ok(x)
}


/// Applies a transformer block to the input tensor `x`.
///
/// A transformer block consists of a multi-head attention layer followed by a
/// feedforward network. The input tensor `x` is first normalized using layer
/// normalization, then passed through the multi-head attention layer. The
/// output is normalized again and passed through the feedforward network. 
/// The outputs of these two layers are added sequentally to the original input
/// `x`.
///
/// Parameters:
/// - `x`: The input tensor of shape `(seq_len, emb_dim)`.
/// - `mlp`: A hashmap containing the weights and biases for the feedforward
/// network.
/// - `attn`: A hashmap containing the weights and biases for the multi-head
/// attention layer.
/// - `ln_1`: A hashmap containing the layer normalization parameters for the 
/// first layer normalization.
/// - `ln_2`: A hashmap containing the layer normalization parameters for the 
/// second layer normalization.
/// - `n_head`: The number of attention heads.
///
/// Returns:
/// The output tensor of shape `(seq_len, emb_dim)`.
fn transformer_block(
    x: Array2<f32>,
    mlp: &HashMap<String, Value>,
    attn: &HashMap<String, Value>,
    ln_1: &HashMap<String, Value>,
    ln_2: &HashMap<String, Value>,
    n_head: i64,
) -> Array2<f32> {
    // TODO: is there a way to remove the clone?
    let normalized_x = layer_norm(x.clone(), &ln_1["g"], &ln_1["b"], 1e-5).unwrap();
    let mha_out = x + mha(normalized_x, &attn["c_attn"], &attn["c_proj"], n_head).unwrap();
    let normalized_mha_out = layer_norm(mha_out.clone(), &ln_2["g"], &ln_2["b"], 1e-5).unwrap();
    let ffn_out = ffn(normalized_mha_out, &mlp["c_fc"], &mlp["c_proj"]).unwrap();
    mha_out + ffn_out
}

/// Applies the GPT-2 model to the given input tokens.
///
/// This function takes a vector of input token IDs, along with various model
/// parameters, and applies the GPT-2 model to generate the output logits.
///
/// Parameters:
/// - `inputs`: A vector of input token IDs.
/// - `wte`: The word token embedding matrix.
/// - `wpe`: The word position embedding matrix.
/// - `blocks`: A vector of transformer block parameters.
/// - `ln_f`: The final layer normalization parameters.
/// - `n_head`: The number of attention heads.
///
/// Returns:
/// The output logits as a 2D array.
fn gpt2(
    inputs: &Vec<i64>,
    wte: &Array2<f32>,
    wpe: &Array2<f32>,
    blocks: &Vec<Value>,
    ln_f: &HashMap<String, Value>,
    n_head: i64,
) -> Result<Array2<f32>, String> {
    // lookup the embeddings for the input tokens in the wte (word token embedding) matrix
    let indices: Vec<usize> = inputs.iter().map(|&x| x as usize).collect();
    let token_embed = wte.select(Axis(0), &indices);
    // lookup the positional encoding for the input tokens in the wpe (word position embedding) matrix
    let indices: Vec<usize> = (0..inputs.len()).collect();
    let position_emb = wpe.select(Axis(0), &indices);
    // combine the token embeddings and positional embeddings
    let mut x = token_embed + position_emb;
    for block in blocks {
        let this_block = match block {
            Value::Hashmap(map) => map,
            other => Err(format!(
                "Expected Hashmap for transformer block, got {:?}",
                other
            ))?,
        };
        let attn = match &this_block["attn"] {
            Value::Hashmap(map) => map,
            other => Err(format!("Expected Hashmap for attn, got {:?}", other))?,
        };
        let mlp = match &this_block["mlp"] {
            Value::Hashmap(map) => map,
            other => Err(format!("Expected Hashmap for mlp, got {:?}", other))?,
        };
        let ln_1 = match &this_block["ln_1"] {
            Value::Hashmap(map) => map,
            other => Err(format!("Expected Hashmap for ln_1, got {:?}", other))?,
        };
        let ln_2 = match &this_block["ln_2"] {
            Value::Hashmap(map) => map,
            other => Err(format!("Expected Hashmap for ln_2, got {:?}", other))?,
        };
        x = transformer_block(x, mlp, attn, ln_1, ln_2, n_head);
    }
    Ok(layer_norm(x, &ln_f["g"], &ln_f["b"], 1e-5)
        .unwrap()
        .dot(&wte.t()))
}

/// Generates a sequence of tokens using a pre-trained GPT-2 model.
///
/// # Arguments
/// - `inputs`: The initial input sequence of tokens.
/// - `params`: A HashMap containing the model parameters, including the
/// word token embeddings, word position embeddings, transformer blocks, 
/// and layer normalization parameters.
/// - `n_head`: The number of attention heads in the transformer blocks.
/// - `n_tokens_to_generate`: The number of tokens to generate.
///
/// # Returns
/// A vector of generated token IDs.
fn generate(
    mut inputs: Vec<i64>,
    params: HashMap<String, Value>,
    n_head: i64,
    n_tokens_to_generate: usize,
) -> Vec<i64> {
    let wte = match &params["wte"] {
        Value::NumpyArray(arr) => arr,
        _ => panic!("Expected NumpyArray"),
    };

    let wpe = match &params["wpe"] {
        Value::NumpyArray(arr) => arr,
        _ => panic!("Expected NumpyArray"),
    };

    let blocks = match &params["blocks"] {
        Value::List(list) => list,
        _ => panic!("Expected List for Blocks"),
    };
    let ln_f = match &params["ln_f"] {
        Value::Hashmap(map) => map,
        _ => {
            // println!("ln_f: {:?}", params["ln_f"]);
            panic!("Expected List for ln_f");
        }
    };

    for _ in 0..n_tokens_to_generate {
        let logits = gpt2(&inputs, wte, wpe, blocks, ln_f, n_head).unwrap();
        // get the last row (i.e. the last token)
        let last_row = logits.row(logits.nrows() - 1);
        // argmax
        let next_id = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        // append prediction to input
        inputs.push(next_id as i64);
    }
    inputs[inputs.len() - n_tokens_to_generate..].to_vec()
}

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    let args = Args::parse();
    println!("args: {:?}", args.prompt);
    let prompt = args.prompt;
    let n_tokens_to_generate = args.n_tokens_to_generate;

    // load encoder, hparams, and params from the released open-ai gpt-2 files
    let py_foo = c_str!(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/utils.py"
    )));
    let from_python: Py<PyAny> = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
        let module = PyModule::from_code(py, py_foo, c_str!("utils.py"), c_str!("utils.py"))?;
        let func = module.getattr("load_encoder_hparams_and_params")?;
        let result = func.call1(("124M", "models"))?;
        Ok(result.into())
    })?;

    // Extract the tuple elements
    let model_params = Python::with_gil(|py| -> PyResult<Params> {
        // Convert Py<PyAny> to a reference with lifetime.
        let tuple = from_python.bind(py);
        // Access tuple elements using indexing.
        let encoder = tuple.get_item(0)?;
        let hparams = tuple.get_item(1)?;
        let params = tuple.get_item(2)?;

        // encode the input string using the BPE tokenizer
        let input_ids = encoder.call_method1("encode", (prompt,))?;
        let input_ids_vec = input_ids.extract()?;

        let new_hparams = extract_hashmap(&hparams);
        let new_params = extract_hashmap(&params);

        let model_params = Params {
            encoder: encoder.into(),
            hparams: new_hparams.unwrap(),
            params: new_params.unwrap(),
            input_ids: input_ids_vec,
        };

        Ok(model_params)
    })?;

    // make sure we we are not suprassing the max sequence length of our model
    let n_ctx = match model_params.hparams["n_ctx"] {
        Value::Int(x) => x,
        _ => panic!("n_ctx is not an integer"),
    };
    let input_ids = model_params.input_ids;
    assert!(input_ids.len() + n_tokens_to_generate < n_ctx as usize);

    let output_ids = generate(input_ids, model_params.params, 12, n_tokens_to_generate);
    Python::with_gil(|py| -> PyResult<_> {
        let encoder = model_params.encoder.bind(py);
        let output_ids = encoder.call_method1("decode", (output_ids,))?;
        println!("output_ids: {:?}", output_ids);
        Ok(())
    })?;
    Ok(())
}
