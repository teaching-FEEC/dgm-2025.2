import torch
import torch.nn as nn
import numpy as np
from compact import compact # Certifique-se de que compact.py está no mesmo diretório ou no caminho do Python

# --- Funções auxiliares para converter PyTorch para formato compatível com shaderutils ---

def extract_conv_info(conv_layer):
    """Extrai pesos e bias de uma camada Conv2d."""
    weights = conv_layer.weight.data.cpu().numpy() # Formato: (out_channels, in_channels, H, W)
    bias = conv_layer.bias.data.cpu().numpy() if conv_layer.bias is not None else None
    # Transpõe para o formato necessário pelo shaderutils (H, W, in_channels, out_channels)
    weights = np.transpose(weights, (2, 3, 1, 0))
    return weights, bias

def extract_prelu_params(prelu_layer):
    """Extrai parâmetros do PReLU se for um único valor ou retorna None para múltiplos."""
    # Para modelos compact, o PReLU geralmente é por canal (num_parameters=num_feat)
    # A implementação GLSL original assume escalar. Vamos lidar com o caso por canal.
    # A função PReLU é: f(x) = max(0, x) + a * min(0, x)
    # Onde 'a' é um vetor de parâmetros (um por canal).
    # No GLSL, isso é mais complexo, pois precisamos de uma textura ou constante para os parâmetros 'a'.
    # Para simplificação, se houver apenas um parâmetro (escalar), usamos diretamente.
    # Se houver múltiplos (por canal), geramos código GLSL que assume os parâmetros
    # estão disponíveis como uma constante ou textura (o que é complicado).
    # Uma abordagem comum é converter o PReLU em uma sequência de operações GLSL:
    # PReLU(x) = max(0, x) + a * min(0, x)
    # GLSL: result = max(vec4(0.0), x) + a_vec * min(vec4(0.0), x)
    # Onde a_vec são os parâmetros PReLU para os canais relevantes.
    # Como lidar com 'a_vec' no GLSL para 24, 35 ou 64 canais é desafiador.
    # O shaderutils original não lida com PReLU diretamente, apenas ReLU (crelu).
    # Vamos adaptar a lógica para gerar GLSL representando PReLU.
    # Para cada canal/processamento GLSL (4 canais por vez), precisamos dos 4 parâmetros PReLU correspondentes.
    # Supondo que a ordem dos parâmetros PReLU no modelo PyTorch seja a mesma que a ordem dos canais na saída da conv.
    prelu_params = prelu_layer.weight.data.cpu().numpy()
    # Retorna o array numpy com todos os parâmetros PReLU (num_parameters,)
    return prelu_params


def generate_shader_for_compact(model: nn.Module, output_file: str, hook="MAIN", desc="CompactModel", when=None, max_bind_num=16):
    """
    Gera um shader GLSL a partir de um modelo compact PyTorch.

    Args:
        model (nn.Module): Instância do modelo PyTorch carregado.
        output_file (str): Nome do arquivo de saída para o shader GLSL.
        hook (str): Textura de entrada hook (padrão "MAIN").
        desc (str): Descrição do shader.
        when (str): Condição para aplicar o shader (opcional).
        max_bind_num (int): Número máximo de texturas que podem ser ligadas por shader.
    """
    layers = list(model.children()) # Obtem as camadas principais (body, upsampler)
    body_layers = list(model.body.children()) # Obtem as camadas dentro de body
    upsampler = model.upsampler # Camada PixelShuffle

    shader_string = ""
    current_texture = hook # Começa com a textura MAIN
    current_layer_idx = 0 # Índice para nomear as texturas intermediárias

    # --- Processar a primeira convolução ---
    first_conv = body_layers[0]
    first_weights, first_bias = extract_conv_info(first_conv)
    layer_name = f"conv_first_{current_layer_idx}"
    shader_string += parse_conv2d_pytorch(first_weights, first_bias, layer_name, current_texture, current_texture, desc + "_FirstConv", when, max_bind_num, input_features_shape=first_conv.in_channels)
    current_texture = layer_name # Atualiza a textura de entrada para a próxima camada
    current_layer_idx += 1

    # --- Processar ativação PReLU após a primeira conv ---
    first_prelu = body_layers[1]
    if isinstance(first_prelu, nn.PReLU):
        prelu_params = extract_prelu_params(first_prelu)
        layer_name = f"prelu_first_{current_layer_idx}"
        shader_string += parse_prelu_pytorch(prelu_params, layer_name, current_texture, current_texture, desc + "_FirstPReLU", when, max_bind_num)
        current_texture = layer_name
        current_layer_idx += 1

    # --- Processar o bloco principal (num_conv vezes Conv + PReLU) ---
    for i in range(2, len(body_layers) - 1, 2): # Começa em 2, pula de 2 em 2 (Conv, PReLU)
        conv_layer = body_layers[i]
        if not isinstance(conv_layer, nn.Conv2d):
            print(f"Warning: Expected Conv2d at index {i}, got {type(conv_layer)}. Skipping.")
            continue

        prelu_layer = body_layers[i+1]
        if not isinstance(prelu_layer, nn.PReLU):
             print(f"Warning: Expected PReLU at index {i+1}, got {type(prelu_layer)}. Skipping pair.")
             continue

        conv_weights, conv_bias = extract_conv_info(conv_layer)
        layer_name = f"conv_main_{current_layer_idx}"
        shader_string += parse_conv2d_pytorch(conv_weights, conv_bias, layer_name, current_texture, current_texture, desc + f"_MainConv_{i//2}", when, max_bind_num, input_features_shape=conv_layer.in_channels)
        current_texture = layer_name
        current_layer_idx += 1

        prelu_params = extract_prelu_params(prelu_layer)
        layer_name = f"prelu_main_{current_layer_idx}"
        shader_string += parse_prelu_pytorch(prelu_params, layer_name, current_texture, current_texture, desc + f"_MainPReLU_{i//2}", when, max_bind_num)
        current_texture = layer_name
        current_layer_idx += 1

    # --- Processar a última convolução (antes do PixelShuffle) ---
    last_conv = body_layers[-1] # Última camada do body
    last_weights, last_bias = extract_conv_info(last_conv)
    layer_name = f"conv_last_{current_layer_idx}"
    shader_string += parse_conv2d_pytorch(last_weights, last_bias, layer_name, current_texture, layer_name, desc + "_LastConv", when, max_bind_num, input_features_shape=last_conv.in_channels) # Salva em uma textura temporária
    current_texture = layer_name
    current_layer_idx += 1

    # --- Processar PixelShuffle ---
    upscale_factor = model.upscale
    layer_name = f"pixel_shuffle_{current_layer_idx}"
    shader_string += parse_pixelshuffle_pytorch(upscale_factor, layer_name, current_texture, hook, desc + "_PixelShuffle", when, max_bind_num)
    # O PixelShuffle salva no hook final (MAIN ou outro) e adiciona a base
    current_texture = hook # O resultado está no hook final
    current_layer_idx += 1

    # --- Salvar o shader gerado ---
    with open(output_file, "w") as f:
        f.write(shader_string)

    print(f"Shader GLSL gerado e salvo em: {output_file}")


def parse_conv2d_pytorch(weights, bias, layer_name, input_texture, save_texture, desc, when, max_bind_num, input_features_shape):
    """Gera shader GLSL para uma camada Conv2d PyTorch.
       Esta função é simplificada para lidar com modelos sequenciais como 'compact'.
       Ela não depende de funções de parsing de árvore de modelo do shaderutils original.
    """
    # Esta função é uma adaptação simplificada de parse_conv2d do shaderutils original
    # Assume kernel 3x3, padding 1, stride 1, e uma única entrada sequencial
    kernel_height, kernel_width, input_size, output_size = weights.shape
    if kernel_height != 3 or kernel_width != 3:
        print(f"Warning: Kernel size {kernel_height}x{kernel_width} not 3x3. Shader might not be optimal.")
    if input_size != input_features_shape:
        print(f"Warning: Declared input features {input_features_shape} != actual input features {input_size} for layer {layer_name}")

    shader_str = ""
    # Para modelos sequenciais, não precisamos dividir em 'binds' complexos.
    # Processamos os canais de saída em blocos de 4 diretamente.
    for n in range(0, output_size, 4): # n: índice do canal de saída inicial do bloco 4x4
        n_i = n // 4
        current_bind_name = save_texture + (str(n_i) if n_i > 0 else "")

        # --- Cabeçalho do shader ---
        shader_str += f"//!DESC {desc}-Conv-{min(4, output_size - n)}x{kernel_height}x{kernel_width}x{input_size}\n"
        shader_str += f"//!HOOK {input_texture}\n"
        shader_str += f"//!BIND {input_texture}\n"
        if save_texture != input_texture:
             shader_str += f"//!BIND {save_texture} # Usado para shaders subsequentes se necessário\n"
        shader_str += f"//!SAVE {current_bind_name}\n"
        shader_str += f"//!WIDTH {input_texture}.w\n"
        shader_str += f"//!HEIGHT {input_texture}.h\n"
        if current_bind_name not in ["LUMA", "NATIVE", "RGB", "MAIN"]:
            shader_str += "//!COMPONENTS 4\n"
        if when:
            shader_str += f"//!WHEN {when}\n"

        # --- Corpo do shader ---
        shader_str += "vec4 hook() {\n"
        init_result = False
        for i, j in np.ndindex((kernel_height, kernel_width)): # i, j: coordenadas do kernel
            tex_off = (float(i - kernel_height // 2 + (1 - kernel_height % 2)), float(j - kernel_width // 2 + (1 - kernel_width % 2)))
            # Para modelos sequenciais, a 'entrada' é sempre a textura 'input_texture'
            # e o número de canais de entrada é 'input_size'
            current_start_feature = 0 # Começa do primeiro canal de entrada
            current_feature_length = input_size # Usa todos os canais de entrada

            # --- Debugging: Verificar valores antes do fatiamento ---
            # print(f"DEBUG: Layer {layer_name}, n={n}, i={i}, j={j}, current_start_feature={current_start_feature}, current_feature_length={current_feature_length}")
            # print(f"DEBUG: weights.shape={weights.shape}, input_size={input_size}, output_size={output_size}")
            if current_start_feature < 0 or current_feature_length < 0 or n < 0 or (current_start_feature + current_feature_length) > weights.shape[2] or n > weights.shape[3]:
                print(f"ERROR: Invalid indices for slicing weights. sf={current_start_feature}, fl={current_feature_length}, n={n}")
                print(f"ERROR: Shape: {weights.shape}, requested: [{i}, {j}, {current_start_feature}:{current_start_feature+current_feature_length}, {n}:?]")
                return "" # Retorna string vazia ou lança um erro para interromper

            #Gather current weights
            # Acessa os pesos para os canais de entrada (0 a input_size-1) e os canais de saída (n a n+4 ou até output_size)
            # CORREÇÃO: O índice final deve ser min(n+4, output_size) para evitar out-of-bounds
            end_idx_out = min(n + 4, output_size)
            current_weights_slice = weights[i, j, current_start_feature:current_start_feature+current_feature_length, n:end_idx_out]
            # print(f"DEBUG: current_weights_slice.shape={current_weights_slice.shape}")

            # --- Debugging: Verificar valores antes do np.pad ---
            pad_h_needed = max(0, 4 - current_weights_slice.shape[0])
            pad_w_needed = max(0, 4 - current_weights_slice.shape[1])
            # print(f"DEBUG: Padding needed: ({pad_h_needed}, {pad_w_needed}) for shape {current_weights_slice.shape}")
            if pad_h_needed < 0 or pad_w_needed < 0:
                 print(f"ERROR: Negative padding calculated: pad_h_needed={pad_h_needed}, pad_w_needed={pad_w_needed}")
                 return ""

            #Pad until 4x4 matrix
            padded_weights = np.pad(current_weights_slice, [[0, pad_h_needed],[0, pad_w_needed]])
            weight_tuple = tuple(padded_weights.flatten())

            shader_str += "    "
            if init_result:
                shader_str += "result += "
            else:
                shader_str += "vec4 result = "
                init_result = True

            if tex_off == (0.0, 0.0):
                 shader_str += f"mat4{weight_tuple} * {input_texture}_tex({input_texture}_pos);\n"
            else:
                 shader_str += f"mat4{weight_tuple} * {input_texture}_texOff(vec2({tex_off[0]}, {tex_off[1]}) * {input_texture}_pt);\n"

        # Adiciona bias se existir (na última iteração de n)
        # CORREÇÃO: O índice final para bias também deve ser min(n+4, output_size)
        if bias is not None and n < len(bias): # Verifica se n está dentro do tamanho do bias
             current_bias_slice = bias[n:min(n+4, len(bias))] # Fatia o bias corretamente
             padded_bias = np.pad(current_bias_slice, [[0, 4 - len(current_bias_slice)]], mode='constant')
             bias_tuple = tuple(padded_bias)
             shader_str += f"    result += vec4{bias_tuple};\n"

        shader_str += "    return result;\n"
        shader_str += "}\n"
    return shader_str


def parse_prelu_pytorch(params, layer_name, input_texture, save_texture, desc, when, max_bind_num):
    """Gera shader GLSL para uma ativação PReLU PyTorch."""
    # Esta função assume PReLU por canal (um parâmetro por canal)
    # O PReLU é: f(x) = max(0, x) + a * min(0, x)
    # Onde 'a' é um vetor de parâmetros aprendidos (um por canal).
    # GLSL: result = max(vec4(0.0), x) + a_vec * min(vec4(0.0), x)
    # Onde a_vec são os 4 parâmetros PReLU correspondentes aos 4 canais processados.

    # Calcula quantos shaders são necessários para cobrir todos os canais de entrada
    num_params = len(params) # Número total de parâmetros PReLU (igual ao número de canais de entrada)
    num_shaders_needed = (num_params + 3) // 4 # Divisão inteira arredondada para cima

    shader_str = ""
    for s_idx in range(num_shaders_needed):
        start_idx = s_idx * 4
        end_idx = min(start_idx + 4, num_params)
        current_params = params[start_idx:end_idx]
        current_params_len = len(current_params)
        # Preenche com zeros se houver menos de 4 parâmetros
        # Corrige o erro: padding não pode ser negativo
        padding_needed = max(0, 4 - current_params_len) # Garante que padding_needed seja >= 0

        # --- Debugging: Verificar valores antes do np.pad em PReLU ---
        # print(f"DEBUG_PReLU: Layer {layer_name}, s_idx={s_idx}, start_idx={start_idx}, end_idx={end_idx}, num_params={num_params}")
        # print(f"DEBUG_PReLU: current_params.shape={current_params.shape}, current_params_len={current_params_len}, padding_needed={padding_needed}")
        if padding_needed < 0:
            print(f"ERROR_PReLU: Negative padding_needed calculated: {padding_needed}")
            return ""

        padded_params = np.pad(current_params, (0, padding_needed), mode='constant')

        layer_name_s = f"{layer_name}_s{s_idx}"
        shader_str += f"//!DESC {desc}-PReLU-Shard{s_idx}\n"
        shader_str += f"//!HOOK {input_texture}\n"
        shader_str += f"//!BIND {input_texture}\n"
        shader_str += f"//!SAVE {layer_name_s}\n"
        shader_str += f"//!WIDTH {input_texture}.w\n"
        shader_str += f"//!HEIGHT {input_texture}.h\n"
        if layer_name_s not in ["LUMA", "NATIVE", "RGB", "MAIN"]:
            shader_str += "//!COMPONENTS 4\n"
        if when:
            shader_str += f"//!WHEN {when}\n"

        shader_str += f"// PReLU parameters (shard {s_idx}): {current_params}\n"
        shader_str += "vec4 hook() {\n"
        shader_str += f"    vec4 input_val = {input_texture}_tex({input_texture}_pos);\n"
        shader_str += f"    vec4 prelu_params = vec4({padded_params[0]}, {padded_params[1]}, {padded_params[2]}, {padded_params[3]});\n"
        shader_str += f"    return max(vec4(0.0), input_val) + prelu_params * min(vec4(0.0), input_val);\n"
        shader_str += "}\n"
        shader_str += "\n" # Linha extra entre shaders
    return shader_str


def parse_pixelshuffle_pytorch(upscale_factor, layer_name, input_texture, output_hook, desc, when, max_bind_num):
    """Gera shader GLSL para uma operação PixelShuffle PyTorch."""
    # Esta implementação é baseada na discussão anterior e tenta ser mais robusta.
    # Assume que a entrada (input_texture) tem C_in = C_out * upscale^2 canais.
    # Ex: C_out=3, upscale=2 -> C_in=12.
    # Gera shaders para reorganizar os canais e aumentar a resolução espacial.

    C_out = 3 # Assume 3 canais de saída (RGB)
    C_in_calc = C_out * upscale_factor * upscale_factor # Calcula C_in esperado

    shader_str = ""
    # --- Geração de shaders para reorganizar os canais ---
    # Gera um shader para cada canal de saída desejado (R, G, B)
    for c_out_idx in range(C_out): # Gera um shader para R, G, B
        layer_name_ch = f"{layer_name}_ch{c_out_idx}"
        shader_str += f"//!DESC {desc}-PixelShuffle-Ch{c_out_idx}\n"
        shader_str += f"//!HOOK {input_texture} # Lê a saída da última conv (C_in canais)\n"
        shader_str += f"//!SAVE {layer_name_ch} # Salva um canal de saída intermediário\n"
        shader_str += f"//!WIDTH {input_texture}.w {upscale_factor} *\n"
        shader_str += f"//!HEIGHT {input_texture}.h {upscale_factor} *\n"
        if when:
            shader_str += f"//!WHEN {when}\n"
        shader_str += "vec4 hook() {\n"
        shader_str += f"    // Calcula a posição na textura de entrada (menor)\n"
        shader_str += f"    vec2 input_pos_f = hook_pos * {input_texture}_pt; // hook_pos é a posição na saída upscale\n"
        shader_str += f"    vec2 input_pos_base = floor(input_pos_f * vec2({1.0/upscale_factor}, {1.0/upscale_factor})) * vec2({1.0/upscale_factor}, {1.0/upscale_factor}) + {input_texture}_pt * 0.5; // Ajuste para centro do pixel\n"
        shader_str += f"    vec2 subpixel_f = fract(input_pos_f * vec2({upscale_factor}, {upscale_factor})); // Frac para subpixel\n"
        shader_str += f"    ivec2 subpixel = ivec2(subpixel_f * vec2({upscale_factor}, {upscale_factor})); // Índice do subpixel (0 ou 1 para upscale=2)\n"
        shader_str += f"    int subpixel_idx = subpixel.y * {upscale_factor} + subpixel.x; // Índice linear do subpixel (0-3 para upscale=2)\n"
        shader_str += f"    // Índice do canal de entrada para este canal de saída (c_out_idx) e subpixel (subpixel_idx)\n"
        shader_str += f"    int channel_idx = {c_out_idx} + subpixel_idx * {C_out}; // c_out_idx + subpixel_offset * C_out\n"
        shader_str += f"    // Calcula o índice do bloco de 4 canais e o índice dentro do bloco\n"
        shader_str += f"    int block_idx = channel_idx / 4;\n"
        shader_str += f"    int intra_block_idx = channel_idx % 4;\n"
        shader_str += f"    // Lê o bloco de 4 canais correspondente\n"
        shader_str += f"    vec4 block = {input_texture}_tex(input_pos_base + vec2(block_idx, 0.0) * {input_texture}_pt); // Lê bloco 'block_idx' na horizontal\n"
        shader_str += f"    // Seleciona o valor dentro do bloco usando swizzling\n"
        shader_str += f"    float output_val;\n"
        # Usar swizzling para selecionar o componente com base em intra_block_idx
        # r=0, g=1, b=2, a=3
        swizzle_map = {0: ".r", 1: ".g", 2: ".b", 3: ".a"}
        shader_str += f"    if(intra_block_idx == 0) {{ output_val = block.r; }}\n"
        shader_str += f"    else if(intra_block_idx == 1) {{ output_val = block.g; }}\n"
        shader_str += f"    else if(intra_block_idx == 2) {{ output_val = block.b; }}\n"
        shader_str += f"    else if(intra_block_idx == 3) {{ output_val = block.a; }}\n"
        shader_str += f"    // Retorna o valor para o canal apropriado, outros canais são 0\n"
        shader_str += f"    vec4 result = vec4(0.0);\n"
        if c_out_idx == 0:
            shader_str += f"    result.r = output_val;\n"
        elif c_out_idx == 1:
            shader_str += f"    result.g = output_val;\n"
        elif c_out_idx == 2:
            shader_str += f"    result.b = output_val;\n"
        # O canal A pode ser zero ou conter o residual, mas o residual é adicionado depois
        shader_str += f"    result.a = 1.0; // Alfa fixo\n"
        shader_str += f"    return result;\n"
        shader_str += f"}}\n"
        shader_str += f"\n" # Linha extra entre shaders

    # --- Shader final para combinar os canais e adicionar o residual ---
    combined_layer_name = f"{layer_name}_combined"
    shader_str += f"//!DESC {desc}-CombineAndResidual\n"
    shader_str += f"//!HOOK {output_hook} # hook_pos e hook_tex são da textura MAIN original\n"
    shader_str += f"//!BIND {output_hook} # Textura original para o residual (base)\n"
    shader_str += f"//!BIND {layer_name}_ch0 # Canal R\n"
    shader_str += f"//!BIND {layer_name}_ch1 # Canal G\n"
    shader_str += f"//!BIND {layer_name}_ch2 # Canal B\n"
    shader_str += f"//!SAVE {output_hook} # Salva de volta na MAIN\n"
    shader_str += f"//!WIDTH {layer_name}_ch0.w # Já está upscale\n"
    shader_str += f"//!HEIGHT {layer_name}_ch0.h\n"
    if when:
        shader_str += f"//!WHEN {when}\n"
    shader_str += "vec4 hook() {\n"
    shader_str += f"    // Parte super-resolucionada (SR)\n"
    shader_str += f"    vec4 sr_part = vec4({layer_name}_ch0_tex({layer_name}_ch0_pos).r,\n"
    shader_str += f"                     {layer_name}_ch1_tex({layer_name}_ch1_pos).g,\n"
    shader_str += f"                     {layer_name}_ch2_tex({layer_name}_ch2_pos).b, 1.0);\n"
    shader_str += f"    // Parte residual (base interpolada)\n"
    shader_str += f"    vec4 base = {output_hook}_tex({output_hook}_pos); // hook_pos é a posição na saída upscale\n"
    shader_str += f"    // Adiciona a parte super-resolucionada ao residual\n"
    shader_str += f"    return sr_part + base;\n"
    shader_str += f"}}\n"

    return shader_str


# --- Exemplo de uso ---
if __name__ == "__main__":
    # --- 1. Instanciar o modelo com os parâmetros corretos ---
    # Exemplo: Super model 1x
    #model = compact(num_in_ch=3, num_out_ch=3, num_feat=24, num_conv=8, upscale=1, act_type="prelu")
    # Exemplo: Mega model 2x
    model = compact(num_in_ch=3, num_out_ch=3, num_feat=35, num_conv=8, upscale=2, act_type="prelu")
    # Exemplo: Ultra model 1x
    #model = compact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=8, upscale=1, act_type="prelu")
    
    print("Estrutura do modelo Super (num_feat=24, num_conv=8, upscale=1):")
    print(model)
    print("\nNúmero total de parâmetros:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # --- 2. Carregar o checkpoint ---
    checkpoint_path = "2x_mega_28i_2X-10.pth" # Substitua pelo caminho do seu modelo

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint carregado de: {checkpoint_path}")
        print(f"Chaves no checkpoint: {list(checkpoint.keys())}") # Verifica as chaves

        # --- 3. Tentar carregar o state_dict ---
        # Procura pelas chaves mais comuns para state_dict
        state_dict_key = None
        if "params" in checkpoint:
            state_dict_key = "params"
        elif "state_dict" in checkpoint:
            state_dict_key = "state_dict"
        elif hasattr(checkpoint, 'keys') and len(list(checkpoint.keys())) == 1:
            # Se houver apenas uma chave, assume que é o state_dict
            state_dict_key = list(checkpoint.keys())[0]
        else:
            # Se nenhuma chave conhecida for encontrada, assume que o próprio checkpoint é o state_dict
            # Isso levantará um erro na próxima linha se for o caso incorreto
            state_dict_key = "direct_load"

        if state_dict_key != "direct_load":
            print(f"Usando chave '{state_dict_key}' para carregar o state_dict.")
            model.load_state_dict(checkpoint[state_dict_key], strict=True) # Use strict=True para garantir correspondência exata
        else:
            print("Tentando carregar o checkpoint diretamente como state_dict.")
            model.load_state_dict(checkpoint, strict=True)

        model.eval() # Coloque em modo de avaliação
        print("Modelo carregado com sucesso.")

        # --- 4. Gerar o shader ---
        # Gera o nome do shader a partir do nome do modelo
        output_shader_name = checkpoint_path.replace('.pth', '.glsl').replace('.pt', '.glsl')
        model_name_part = checkpoint_path.split('_')[0].split('/')[-1].upper() # Extrai "1x", "2x", etc., ou o nome base
        upscale_part = checkpoint_path.split('_')[1].split('.')[0].upper() # Extrai "super", "mega", "ultra"
        desc_label = f"Compact_{upscale_part.upper()}_{upscale_part}" # Ex: Compact_SUPER_1x

        generate_shader_for_compact(
            model,
            output_file=output_shader_name,
            hook="MAIN",
            desc=desc_label,
            when=f"OUTPUT.w {model.upscale} * MAIN.w > OUTPUT.h {model.upscale} * MAIN.h >" # Ex: Aplica se a saída for maior que a entrada * upscale
        )

    except FileNotFoundError:
        print(f"Arquivo de modelo {checkpoint_path} não encontrado.")
    except KeyError as e:
        print(f"Chave não encontrada no checkpoint: {e}")
        print("Verifique se o nome do modelo e a estrutura do checkpoint estão corretos.")
    except RuntimeError as e:
        print(f"Erro ao carregar o modelo: {e}")
        print("Verifique se os parâmetros do modelo (num_feat, num_conv, upscale) correspondem aos usados durante o treinamento.")
    except Exception as e:
        print(f"Erro inesperado: {e}")
