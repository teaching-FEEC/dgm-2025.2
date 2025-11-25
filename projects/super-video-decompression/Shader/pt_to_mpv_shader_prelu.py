import os
import torch
import torch.nn as nn
import numpy as np

try:
    from compact import compact  # Tenta importar o módulo 'compact'
except ImportError:
    print("AVISO: Módulo 'compact.py' não encontrado. O script só funcionará se você fornecer um que possa ser importado.")
    class compact(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("ERRO: Módulo 'compact' real não encontrado. A instanciação do modelo falhará.")
            self.body = nn.Sequential(nn.Identity())
            self.upsampler = nn.Identity()
            self.upscale = 1
            self.num_feat = 0
            self.num_conv = 0
        def forward(self, x): return x

# --- Funções auxiliares ---

def extract_conv_info(conv_layer):
    weights = conv_layer.weight.data.cpu().numpy()
    bias = conv_layer.bias.data.cpu().numpy() if conv_layer.bias is not None else None
    weights = np.transpose(weights, (2, 3, 1, 0))
    return weights, bias

def extract_prelu_params(prelu_layer):
    return prelu_layer.weight.data.cpu().numpy()

def format_float(f):
    return f"{float(f):.16g}"

# --- Inferência de Parâmetros (sem alterações) ---

def get_compact_params_from_instance(model_instance: nn.Module):
    if hasattr(model_instance, 'num_feat') and hasattr(model_instance, 'num_conv') and hasattr(model_instance, 'upscale'):
        return model_instance.num_feat, model_instance.num_conv, model_instance.upscale
    else:
        raise AttributeError(f"A instância do modelo {type(model_instance)} não possui os atributos esperados (num_feat, num_conv, upscale).")

def get_compact_params_from_state_dict(state_dict, prefix=""):
    relevant_keys = [k for k in state_dict.keys() if k.startswith(prefix)]
    if not relevant_keys: return None, None, None
    body_keys = [k for k in relevant_keys if k.startswith(f"{prefix}body.")]
    if not body_keys: return None, None, None
    layer_map = {}
    for key in body_keys:
        parts = key.split('.'); body_idx_start = -1
        for i, part in enumerate(parts):
            if part == 'body' and i < len(parts) - 1:
                try:
                    layer_idx = int(parts[i+1]); body_idx_start = i+1; break
                except ValueError: continue
        if body_idx_start != -1:
            layer_idx = int(parts[body_idx_start])
            if layer_idx not in layer_map: layer_map[layer_idx] = []
            layer_map[layer_idx].append(key)
    if not layer_map: return None, None, None
    max_layer_idx = max(layer_map.keys())
    conv_weight_keys_max = [k for k in layer_map[max_layer_idx] if 'weight' in k and not any(part in k for part in ['prelu', 'bias'])]
    if not conv_weight_keys_max: return None, None, None
    full_key_max = f"{prefix}{conv_weight_keys_max[0]}" if prefix else conv_weight_keys_max[0]
    if full_key_max not in state_dict: return None, None, None
    conv_weight_tensor_max = state_dict[full_key_max]
    if conv_weight_tensor_max.ndim != 4: return None, None, None
    out_ch_max = conv_weight_tensor_max.shape[0]
    upscale_f = np.sqrt(out_ch_max / 3)
    if not np.isclose(upscale_f, int(upscale_f)): return None, None, None
    upscale = int(upscale_f)
    if upscale < 1 or upscale > 4: return None, None, None
    inferred_num_feat = conv_weight_tensor_max.shape[1]
    inferred_num_conv = (max_layer_idx - 2) // 2
    prev_idx = max_layer_idx - 1
    if prev_idx in layer_map:
        prev_keys = layer_map[prev_idx]; prev_weight_keys = [k for k in prev_keys if 'weight' in k]
        if prev_weight_keys:
            full_prev_key = f"{prefix}{prev_weight_keys[0]}" if prefix else prev_weight_keys[0]
            if full_prev_key in state_dict:
                prev_weight_tensor = state_dict[full_prev_key]
                if prev_weight_tensor.ndim != 1: return None, None, None
            else: return None, None, None
        else: return None, None, None
    else: return None, None, None
    return inferred_num_feat, inferred_num_conv, upscale


# --- Funções de Parse Corrigidas (Produção) ---

def parse_conv2d_pytorch(weights, bias, layer_name, input_textures, save_texture, desc, max_bind_num, input_features_shape):
    """
    Gera shaders GLSL para uma camada Conv2D, usando NAME_texOff para leitura de texel.

    Args:
        weights (np.ndarray): Pesos da convolução (kernel_h, kernel_w, in_ch, out_ch).
        bias (np.ndarray or None): Bias da convolução.
        layer_name (str): Nome base da camada.
        input_textures (list of str): Lista de nomes das texturas de entrada (shards).
        save_texture (str): Nome base para salvar as saídas.
        desc (str): Descrição para o shader.
        max_bind_num (int): Número máximo de binds (não utilizado diretamente aqui).
        input_features_shape (int): Número de canais de entrada esperado.

    Returns:
        tuple: (shader_string, saved_outputs_list)
    """
    kernel_height, kernel_width, input_size, output_size = weights.shape

    if input_size != input_features_shape:
        print(f"ALERTA GRAVE: Inconsistência de forma em {layer_name}. "
              f"Shape dos pesos tem {input_size} canais de entrada, "
              f"mas a camada anterior declarou {input_features_shape}. Usando {input_size}.")

    num_input_shards = len(input_textures)
    shader_str = ""
    saved_outputs = []

    for n in range(0, output_size, 4):
        n_i = n // 4
        current_output_shard_name = f"{save_texture}{n_i if n_i > 0 else ''}"
        saved_outputs.append(current_output_shard_name)

        shader_str += f"//!DESC {desc}-Conv-{min(4, output_size - n)}x{kernel_height}x{kernel_width}x{input_size}\n"
        shader_str += f"//!HOOK {input_textures[0]}\n" # Hook na primeira textura de entrada

        for input_shard_name in input_textures:
            shader_str += f"//!BIND {input_shard_name}\n"

        shader_str += f"//!SAVE {current_output_shard_name}\n"
        shader_str += f"//!WIDTH {input_textures[0]}.w\n"  # Usa a largura da primeira entrada
        shader_str += f"//!HEIGHT {input_textures[0]}.h\n" # Usa a altura da primeira entrada
        if current_output_shard_name not in ["LUMA", "NATIVE", "RGB", "MAIN"]:
            shader_str += "//!COMPONENTS 4\n"

        # --- GERAÇÃO DAS MACROS texOff ---
        macro_definitions = []
        macro_usage_order = []
        for i, j in np.ndindex((kernel_height, kernel_width)):
            tex_off = (float(i - kernel_height // 2 + (1 - kernel_height % 2)),
                       float(j - kernel_width // 2 + (1 - kernel_width % 2)))

            # Define uma macro para este deslocamento específico
            # O nome da macro pode incluir o índice e o deslocamento para garantir unicidade
            macro_name = f"tex_{layer_name}_off_{i}_{j}"
            macro_definitions.append(f"#define {macro_name}(input_tex_name) (input_tex_name##_texOff(vec2({format_float(tex_off[0])}, {format_float(tex_off[1])})))")
            macro_usage_order.append(macro_name)

        # Adiciona as macros ao shader
        for macro_def in macro_definitions:
            shader_str += f"{macro_def}\n"

        shader_str += "vec4 hook() {\n"
        shader_str += "    vec4 result = vec4(0.0);\n"

        for idx, (i, j) in enumerate(np.ndindex((kernel_height, kernel_width))):
            macro_name = macro_usage_order[idx]

            for s_idx in range(num_input_shards):
                start_in_ch = s_idx * 4
                end_in_ch = min(start_in_ch + 4, input_size)
                start_out_ch = n
                end_out_ch = min(start_out_ch + 4, output_size)

                current_weights_slice = weights[i, j, start_in_ch:end_in_ch, start_out_ch:end_out_ch]

                pad_h_needed = 4 - current_weights_slice.shape[0]
                pad_w_needed = 4 - current_weights_slice.shape[1]

                padded_weights = np.pad(current_weights_slice,
                                        [[0, pad_h_needed], [0, pad_w_needed]],
                                        mode='constant')

                weight_strings = [format_float(w) for w in padded_weights.flatten(order='F')]
                weight_tuple_str = f"({', '.join(weight_strings)})"

                input_shard_name = input_textures[s_idx]
                # Usa a macro gerada
                tex_read_str = f"{macro_name}({input_shard_name})"
                shader_str += f"    result += mat4{weight_tuple_str} * {tex_read_str};\n"

        if bias is not None and n < len(bias):
            current_bias_slice = bias[n:min(n+4, len(bias))]
            padded_bias = np.pad(current_bias_slice, [[0, 4 - len(current_bias_slice)]], mode='constant')
            bias_strings = [format_float(b) for b in padded_bias]
            bias_tuple_str = f"({', '.join(bias_strings)})"
            shader_str += f"    result += vec4{bias_tuple_str};\n"

        shader_str += "    return result;\n"
        shader_str += "}\n\n"
    return shader_str, saved_outputs


def parse_prelu_pytorch(params, layer_name, input_textures, save_texture, desc, max_bind_num):
    """
    Gera shaders GLSL para uma camada PReLU.

    Args:
        params (np.ndarray): Parâmetros do PReLU.
        layer_name (str): Nome base da camada.
        input_textures (list of str): Lista de nomes das texturas de entrada (shards).
        save_texture (str): Nome base para salvar as saídas.
        desc (str): Descrição para o shader.
        max_bind_num (int): Número máximo de binds (não utilizado diretamente aqui).

    Returns:
        tuple: (shader_string, saved_outputs_list)
    """
    num_params = len(params)
    num_output_shards = (num_params + 3) // 4
    shader_str = ""
    saved_outputs = []

    # Cada estágio de PReLU lê de um shard de entrada correspondente
    # Se houver mais shards de PReLU que de entrada, reutiliza o último shard de entrada
    for s_idx in range(num_output_shards):
        start_idx = s_idx * 4
        end_idx = min(start_idx + 4, num_params)
        current_params = params[start_idx:end_idx]
        padding_needed = max(0, 4 - len(current_params))
        padded_params = np.pad(current_params, (0, padding_needed), mode='constant')

        # Determina qual shard de entrada usar
        input_shard_idx = min(s_idx, len(input_textures) - 1) # Reutiliza o último shard se necessário
        input_shard_name = input_textures[input_shard_idx]

        output_shard_name = f"{save_texture}{s_idx if s_idx > 0 else ''}"
        saved_outputs.append(output_shard_name)

        shader_str += f"//!DESC {desc}-PReLU-Shard{s_idx}\n"
        shader_str += f"//!HOOK {input_shard_name}\n" # Hook no shard de entrada correspondente
        shader_str += f"//!BIND {input_shard_name}\n"
        shader_str += f"//!SAVE {output_shard_name}\n"
        shader_str += f"//!WIDTH {input_shard_name}.w\n"
        shader_str += f"//!HEIGHT {input_shard_name}.h\n"
        if output_shard_name not in ["LUMA", "NATIVE", "RGB", "MAIN"]:
            shader_str += "//!COMPONENTS 4\n"

        param_strings = [format_float(p) for p in padded_params]
        prelu_params_str = f"vec4({', '.join(param_strings)})"

        shader_str += f"// PReLU parameters (shard {s_idx}): {current_params}\n"
        shader_str += "vec4 hook() {\n"
        # Leitura direta do texel (não usa desvios)
        shader_str += f"    vec4 input_val = {input_shard_name}_tex({input_shard_name}_pos);\n"
        shader_str += f"    vec4 prelu_params = {prelu_params_str};\n"
        shader_str += f"    return max(vec4(0.0), input_val) + prelu_params * min(vec4(0.0), input_val);\n"
        shader_str += "}\n\n"

    return shader_str, saved_outputs


def parse_pixelshuffle_pytorch(upscale_factor, layer_name, input_textures, output_hook, desc, max_bind_num):
    """
    Gera shaders GLSL para a operação Pixel Shuffle e combinação com a entrada original.

    Args:
        upscale_factor (int): Fator de upscale (1, 2, 3, 4).
        layer_name (str): Nome base da camada.
        input_textures (list of str): Lista de nomes das texturas de entrada (shards da última conv).
        output_hook (str): Textura final para salvar o resultado (ex: MAIN).
        desc (str): Descrição para o shader.
        max_bind_num (int): Número máximo de binds (não utilizado diretamente aqui).

    Returns:
        tuple: (shader_string, saved_outputs_list) -> A lista de saída é vazia, pois o resultado é salvo em output_hook.
    """
    C_out = 3
    C_in_calc = C_out * upscale_factor * upscale_factor
    num_input_shards = len(input_textures)

    shader_str = ""

    # Caso especial: upscale=1 (sem upscaling, apenas extração direta dos 3 canais RGB)
    if upscale_factor == 1:
        shader_str += f"//!DESC {desc}-DirectOutput\n"
        shader_str += f"//!HOOK {output_hook}\n"
        shader_str += f"//!BIND {output_hook}\n"
        for input_shard_name in input_textures:
            shader_str += f"//!BIND {input_shard_name}\n"
        shader_str += f"//!SAVE {output_hook}\n"
        shader_str += f"//!WIDTH {input_textures[0]}.w\n"
        shader_str += f"//!HEIGHT {input_textures[0]}.h\n"
        
        shader_str += "vec4 hook() {\n"
        shader_str += f"    vec4 base = {output_hook}_tex({output_hook}_pos);\n"
        shader_str += f"    vec3 sr_part = vec3({input_textures[0]}_tex({input_textures[0]}_pos).rgb);\n"
        shader_str += f"    return vec4(base.rgb + sr_part, 1.0);\n"
        shader_str += "}\n\n"
        
        return shader_str, []

    # --- Gera shaders para cada canal de saída do Pixel Shuffle ---
    output_shard_names = []
    for c_out_idx in range(C_out):
        layer_name_ch = f"{layer_name}_ch{c_out_idx}"
        output_shard_names.append(layer_name_ch)

        shader_str += f"//!DESC {desc}-PixelShuffle-Ch{c_out_idx}\n"
        shader_str += f"//!HOOK {input_textures[0]}\n" # Hook na primeira textura de entrada

        for input_shard_name in input_textures:
            shader_str += f"//!BIND {input_shard_name}\n"

        shader_str += f"//!SAVE {layer_name_ch}\n"
        shader_str += f"//!WIDTH {input_textures[0]}.w {upscale_factor} *\n"  # Escala a largura
        shader_str += f"//!HEIGHT {input_textures[0]}.h {upscale_factor} *\n" # Escala a altura
        shader_str += "//!COMPONENTS 1\n" # Cada canal de saída é 1 componente

        shader_str += "vec4 hook() {\n"
        # Corrige o cálculo da posição e subpixel para o upscale
        shader_str += f"    vec2 input_pos_f = hook_pos * {input_textures[0]}_pt / float({upscale_factor});\n" # hook_pos é a posição no output alvo
        shader_str += f"    vec2 input_pos_base_f = floor(input_pos_f) + 0.5;\n" # Coordenada do texel no input
        shader_str += f"    ivec2 subpixel_int = ivec2(mod(floor(hook_pos * {input_textures[0]}_pt), vec2({float(upscale_factor)})));\n"
        shader_str += f"    int subpixel_idx = subpixel_int.y * {upscale_factor} + subpixel_int.x;\n"
        shader_str += f"    int channel_idx = {c_out_idx} + subpixel_idx * {C_out};\n"
        shader_str += f"    int block_idx = channel_idx / 4;\n"
        shader_str += f"    int intra_block_idx = channel_idx % 4;\n"

        # Lê do shard correto e índice dentro do shard usando uma cadeia de if/else
        # O índice do shard é limitado ao número total de shards
        shader_str += f"    vec4 block;\n"
        shader_str += f"    if (block_idx >= {len(input_textures)}) {{\n"
        shader_str += f"        block = {input_textures[-1]}_tex(input_pos_base_f); // Reutiliza o último shard se o índice for inválido\n"
        shader_str += f"    }}\n"
        for idx, input_shard_name in enumerate(input_textures):
            if idx == 0:
                shader_str += f"    else if (block_idx == {idx}) {{\n"
            else:
                shader_str += f"    else if (block_idx == {idx}) {{\n"
            shader_str += f"        block = {input_shard_name}_tex(input_pos_base_f);\n"
            shader_str += f"    }}\n"

        shader_str += f"    float output_val = 0.0;\n"
        shader_str += f"    if(intra_block_idx == 0) {{ output_val = block.r; }}\n"
        shader_str += f"    else if(intra_block_idx == 1) {{ output_val = block.g; }}\n"
        shader_str += f"    else if(intra_block_idx == 2) {{ output_val = block.b; }}\n"
        shader_str += f"    else if(intra_block_idx == 3) {{ output_val = block.a; }}\n"
        shader_str += f"    return vec4(output_val, 0.0, 0.0, 1.0);\n" # Retorna como componente R
        shader_str += f"}}\n\n"

    # --- Combina os canais SR com a entrada original (MAIN) ---
    shader_str += f"//!DESC {desc}-CombineAndResidual\n"
    shader_str += f"//!HOOK {output_hook}\n" # Hook na textura final (MAIN)
    shader_str += f"//!BIND {output_hook}\n" # Bind da entrada original (MAIN)
    for ch_name in output_shard_names:
        shader_str += f"//!BIND {ch_name}\n" # Bind dos canais SR

    shader_str += f"//!SAVE {output_hook}\n" # Salva de volta em MAIN
    shader_str += f"//!WIDTH {output_shard_names[0]}.w\n" # Usa as dimensões dos canais SR escalados
    shader_str += f"//!HEIGHT {output_shard_names[0]}.h\n"

    shader_str += "vec4 hook() {\n"
    shader_str += f"    vec4 sr_part = vec4({output_shard_names[0]}_tex({output_shard_names[0]}_pos).r,\n"
    shader_str += f"                         {output_shard_names[1]}_tex({output_shard_names[1]}_pos).r,\n"
    shader_str += f"                         {output_shard_names[2]}_tex({output_shard_names[2]}_pos).r, 1.0);\n"
    shader_str += f"    vec4 base = {output_hook}_tex({output_hook}_pos);\n" # Lê a entrada original (MAIN), interpolada para o novo tamanho
    shader_str += f"    return sr_part + base; // Soma a parte super-resolvida com a entrada original\n"
    shader_str += f"}}\n\n"

    return shader_str, [] # Retorna uma lista vazia, pois o estágio final salva em output_hook


# --- Função de Geração Corrigida ---

def generate_shader_for_compact(model: nn.Module, output_file: str, hook="MAIN", desc="CompactModel", max_bind_num=16, start_layer_idx=0):
    body_layers = list(model.body.children())
    shader_string = ""
    # Inicia com a textura base como uma lista de uma única textura
    current_textures = [hook]
    current_layer_idx = start_layer_idx

    # --- Primeira Conv ---
    first_conv = body_layers[0]
    if not isinstance(first_conv, nn.Conv2d): raise TypeError(f"body.0 não é Conv2d")
    first_weights, first_bias = extract_conv_info(first_conv)
    layer_name = f"conv_first_{current_layer_idx}"
    shader_part, current_textures = parse_conv2d_pytorch(
        first_weights, first_bias, layer_name, current_textures, layer_name,
        desc + "_FirstConv", max_bind_num, input_features_shape=first_conv.in_channels
    )
    shader_string += shader_part
    current_layer_idx += 1

    # --- Primeiro PReLU ---
    first_prelu = body_layers[1]
    if isinstance(first_prelu, nn.PReLU):
        prelu_params = extract_prelu_params(first_prelu)
        layer_name = f"prelu_first_{current_layer_idx}"
        shader_part, current_textures = parse_prelu_pytorch(
            prelu_params, layer_name, current_textures, layer_name, desc + "_FirstPReLU", max_bind_num
        )
        shader_string += shader_part
        current_layer_idx += 1

    current_num_features = first_conv.out_channels

    # --- Loop Principal ---
    for i in range(2, len(body_layers) - 1, 2):
        conv_layer = body_layers[i]
        if not isinstance(conv_layer, nn.Conv2d): continue
        prelu_layer = body_layers[i+1]
        if not isinstance(prelu_layer, nn.PReLU): continue

        # Conv
        conv_weights, conv_bias = extract_conv_info(conv_layer)
        layer_name = f"conv_main_{current_layer_idx}"
        shader_part, current_textures = parse_conv2d_pytorch(
            conv_weights, conv_bias, layer_name, current_textures, layer_name,
            desc + f"_MainConv_{i//2}", max_bind_num, input_features_shape=current_num_features
        )
        shader_string += shader_part
        current_layer_idx += 1
        current_num_features = conv_layer.out_channels

        # PReLU
        prelu_params = extract_prelu_params(prelu_layer)
        layer_name = f"prelu_main_{current_layer_idx}"
        shader_part, current_textures = parse_prelu_pytorch(
            prelu_params, layer_name, current_textures, layer_name, desc + f"_MainPReLU_{i//2}", max_bind_num
        )
        shader_string += shader_part
        current_layer_idx += 1

    # --- Última Conv ---
    last_conv = body_layers[-1]
    if not isinstance(last_conv, nn.Conv2d): raise TypeError(f"Última camada do body não é Conv2d")
    last_weights, last_bias = extract_conv_info(last_conv)
    layer_name = f"conv_last_{current_layer_idx}"
    shader_part, current_textures = parse_conv2d_pytorch(
        last_weights, last_bias, layer_name, current_textures, layer_name,
        desc + "_LastConv", max_bind_num, input_features_shape=current_num_features
    )
    shader_string += shader_part
    current_layer_idx += 1

    # --- Pixel Shuffle ---
    upscale_factor = model.upscale
    layer_name = f"pixel_shuffle_{current_layer_idx}"
    shader_part, _ = parse_pixelshuffle_pytorch(
        upscale_factor, layer_name, current_textures, hook, desc + "_PixelShuffle", max_bind_num
    )
    shader_string += shader_part

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(shader_string)
    print(f"Shader de PRODUÇÃO gerado e salvo em: {output_file}")


# Função Principal
def process_all_pth_in_folder(folder_path):
    pth_files = [f for f in os.listdir(folder_path) if f.endswith(('.pth', '.pt'))]
    if not pth_files:
        print(f"Nenhum arquivo .pth ou .pt encontrado na pasta {folder_path}.")
        return
    print(f"Encontrados {len(pth_files)} arquivos .pth/.pt para processar.")

    for pth_file in pth_files:
        checkpoint_path = os.path.join(folder_path, pth_file)
        print(f"\n--- Processando {pth_file} ---")
        try:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            except Exception as e:
                print(f"Falha no carregamento. Tentando com 'weights_only=True'. Erro: {e}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print(f"Checkpoint carregado de: {checkpoint_path}")

            state_dict = None
            if isinstance(checkpoint, dict):
                if 'params' in checkpoint: state_dict = checkpoint['params']
                elif 'params_ema' in checkpoint: state_dict = checkpoint['params_ema']
                else:
                    potential_keys = list(checkpoint.keys())
                    if potential_keys and any(k.startswith('body.') for k in potential_keys):
                        state_dict = checkpoint
                    else:
                        for key, value in checkpoint.items():
                            if isinstance(value, dict) and any(k.startswith('body.') for k in value.keys()):
                                state_dict = value; break
                if state_dict is None: raise KeyError("Estrutura do checkpoint não reconhecida.")
            else: raise ValueError("Formato do checkpoint não suportado.")
            if state_dict is None: raise ValueError("Não foi possível extrair o state_dict.")

            num_feat, num_conv, upscale = get_compact_params_from_state_dict(state_dict, prefix="")
            if num_feat is None or num_conv is None or upscale is None:
                print("Erro: Não foi possível inferir parâmetros do modelo a partir do state_dict.")
                continue

            print(f"Parâmetros inferidos: num_feat={num_feat}, num_conv={num_conv}, upscale={upscale}")

            model_instance = compact(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=upscale, act_type="prelu")
            model_instance.load_state_dict(state_dict, strict=True)
            model_instance.eval()
            print("Modelo carregado com sucesso.")

            output_shader_name = os.path.splitext(checkpoint_path)[0] + '.glsl'
            base_name = os.path.basename(checkpoint_path)
            model_name_part = os.path.splitext(base_name)[0].upper()
            desc_label = f"Compact_{model_name_part}"

            generate_shader_for_compact(
                model_instance,
                output_file=output_shader_name,
                hook="MAIN",
                desc=desc_label,
            )

        except Exception as e:
            print(f"Erro inesperado ao processar {pth_file}: {e}")
            import traceback
            traceback.print_exc()

# --- Execução Principal ---
if __name__ == "__main__":
    folder_path = "."  # Pasta atual
    process_all_pth_in_folder(folder_path)