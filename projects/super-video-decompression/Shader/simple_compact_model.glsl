//!DESC Compact_SIMPLE_COMPACT_MODEL_FirstConv-Conv-4x3x3x3
//!HOOK MAIN
//!BIND MAIN
//!SAVE conv_first_0
//!WIDTH MAIN.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
#define tex_conv_first_0_off_0_0(input_tex_name) (input_tex_name##_texOff(vec2(-1, -1)))
#define tex_conv_first_0_off_0_1(input_tex_name) (input_tex_name##_texOff(vec2(-1, 0)))
#define tex_conv_first_0_off_0_2(input_tex_name) (input_tex_name##_texOff(vec2(-1, 1)))
#define tex_conv_first_0_off_1_0(input_tex_name) (input_tex_name##_texOff(vec2(0, -1)))
#define tex_conv_first_0_off_1_1(input_tex_name) (input_tex_name##_texOff(vec2(0, 0)))
#define tex_conv_first_0_off_1_2(input_tex_name) (input_tex_name##_texOff(vec2(0, 1)))
#define tex_conv_first_0_off_2_0(input_tex_name) (input_tex_name##_texOff(vec2(1, -1)))
#define tex_conv_first_0_off_2_1(input_tex_name) (input_tex_name##_texOff(vec2(1, 0)))
#define tex_conv_first_0_off_2_2(input_tex_name) (input_tex_name##_texOff(vec2(1, 1)))
vec4 hook() {
    vec4 result = vec4(0.0);
    result += mat4(0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, 0, 0, 0, 0, 0, 0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0) * tex_conv_first_0_off_0_0(MAIN);
    result += mat4(0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0, 0, 0, 0, 0, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0) * tex_conv_first_0_off_0_1(MAIN);
    result += mat4(0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, 0, 0, 0, 0, 0, 0, 0, 0) * tex_conv_first_0_off_0_2(MAIN);
    result += mat4(0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0, 0, 0, 0, 0, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0) * tex_conv_first_0_off_1_0(MAIN);
    result += mat4(0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0) * tex_conv_first_0_off_1_1(MAIN);
    result += mat4(0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0, 0, 0, 0, 0, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0) * tex_conv_first_0_off_1_2(MAIN);
    result += mat4(0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, 0, 0, 0, 0, 0, 0, 0, 0, 0) * tex_conv_first_0_off_2_0(MAIN);
    result += mat4(0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0, 0, 0, 0, 0, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0) * tex_conv_first_0_off_2_1(MAIN);
    result += mat4(0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, 0, 0, 0, 0, 0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0) * tex_conv_first_0_off_2_2(MAIN);
    result += vec4(0.009999999776482582, 0.01999999955296516, 0.02999999932944775, 0.03999999910593033);
    return result;
}

//!DESC Compact_SIMPLE_COMPACT_MODEL_FirstPReLU-PReLU-Shard0
//!HOOK conv_first_0
//!BIND conv_first_0
//!SAVE prelu_first_1
//!WIDTH conv_first_0.w
//!HEIGHT conv_first_0.h
//!COMPONENTS 4
// PReLU parameters (shard 0): [0.25 0.5  0.75 1.  ]
vec4 hook() {
    vec4 input_val = conv_first_0_tex(conv_first_0_pos);
    vec4 prelu_params = vec4(0.25, 0.5, 0.75, 1);
    return max(vec4(0.0), input_val) + prelu_params * min(vec4(0.0), input_val);
}

//!DESC Compact_SIMPLE_COMPACT_MODEL_MainConv_1-Conv-4x3x3x4
//!HOOK prelu_first_1
//!BIND prelu_first_1
//!SAVE conv_main_2
//!WIDTH prelu_first_1.w
//!HEIGHT prelu_first_1.h
//!COMPONENTS 4
#define tex_conv_main_2_off_0_0(input_tex_name) (input_tex_name##_texOff(vec2(-1, -1)))
#define tex_conv_main_2_off_0_1(input_tex_name) (input_tex_name##_texOff(vec2(-1, 0)))
#define tex_conv_main_2_off_0_2(input_tex_name) (input_tex_name##_texOff(vec2(-1, 1)))
#define tex_conv_main_2_off_1_0(input_tex_name) (input_tex_name##_texOff(vec2(0, -1)))
#define tex_conv_main_2_off_1_1(input_tex_name) (input_tex_name##_texOff(vec2(0, 0)))
#define tex_conv_main_2_off_1_2(input_tex_name) (input_tex_name##_texOff(vec2(0, 1)))
#define tex_conv_main_2_off_2_0(input_tex_name) (input_tex_name##_texOff(vec2(1, -1)))
#define tex_conv_main_2_off_2_1(input_tex_name) (input_tex_name##_texOff(vec2(1, 0)))
#define tex_conv_main_2_off_2_2(input_tex_name) (input_tex_name##_texOff(vec2(1, 1)))
vec4 hook() {
    vec4 result = vec4(0.0);
    result += mat4(0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0, 0, 0, 0, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161) * tex_conv_main_2_off_0_0(prelu_first_1);
    result += mat4(0, 0, 0, 0, 0, 0, 0, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, 0, 0, 0) * tex_conv_main_2_off_0_1(prelu_first_1);
    result += mat4(-0.1000000014901161, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, 0, 0, 0, 0, 0, 0, 0, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161) * tex_conv_main_2_off_0_2(prelu_first_1);
    result += mat4(0, 0, 0, 0, 0, 0, 0, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, 0, 0, 0) * tex_conv_main_2_off_1_0(prelu_first_1);
    result += mat4(0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0.300000011920929, 0.300000011920929, 0.300000011920929, 0.300000011920929, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161) * tex_conv_main_2_off_1_1(prelu_first_1);
    result += mat4(0, 0, 0, 0, 0, 0, 0, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, 0, 0, 0) * tex_conv_main_2_off_1_2(prelu_first_1);
    result += mat4(-0.1000000014901161, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, 0, 0, 0, 0, 0, 0, 0, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161) * tex_conv_main_2_off_2_0(prelu_first_1);
    result += mat4(0, 0, 0, 0, 0, 0, 0, 0, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0, 0, 0, 0) * tex_conv_main_2_off_2_1(prelu_first_1);
    result += mat4(0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.1000000014901161, 0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 0, 0, 0, 0, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161, -0.1000000014901161) * tex_conv_main_2_off_2_2(prelu_first_1);
    result += vec4(0.05000000074505806, 0.05999999865889549, 0.07000000029802322, 0.07999999821186066);
    return result;
}

//!DESC Compact_SIMPLE_COMPACT_MODEL_MainPReLU_1-PReLU-Shard0
//!HOOK conv_main_2
//!BIND conv_main_2
//!SAVE prelu_main_3
//!WIDTH conv_main_2.w
//!HEIGHT conv_main_2.h
//!COMPONENTS 4
// PReLU parameters (shard 0): [0.9 0.8 0.7 0.6]
vec4 hook() {
    vec4 input_val = conv_main_2_tex(conv_main_2_pos);
    vec4 prelu_params = vec4(0.8999999761581421, 0.800000011920929, 0.699999988079071, 0.6000000238418579);
    return max(vec4(0.0), input_val) + prelu_params * min(vec4(0.0), input_val);
}

//!DESC Compact_SIMPLE_COMPACT_MODEL_LastConv-Conv-3x3x3x4
//!HOOK prelu_main_3
//!BIND prelu_main_3
//!SAVE conv_last_4
//!WIDTH prelu_main_3.w
//!HEIGHT prelu_main_3.h
//!COMPONENTS 4
#define tex_conv_last_4_off_0_0(input_tex_name) (input_tex_name##_texOff(vec2(-1, -1)))
#define tex_conv_last_4_off_0_1(input_tex_name) (input_tex_name##_texOff(vec2(-1, 0)))
#define tex_conv_last_4_off_0_2(input_tex_name) (input_tex_name##_texOff(vec2(-1, 1)))
#define tex_conv_last_4_off_1_0(input_tex_name) (input_tex_name##_texOff(vec2(0, -1)))
#define tex_conv_last_4_off_1_1(input_tex_name) (input_tex_name##_texOff(vec2(0, 0)))
#define tex_conv_last_4_off_1_2(input_tex_name) (input_tex_name##_texOff(vec2(0, 1)))
#define tex_conv_last_4_off_2_0(input_tex_name) (input_tex_name##_texOff(vec2(1, -1)))
#define tex_conv_last_4_off_2_1(input_tex_name) (input_tex_name##_texOff(vec2(1, 0)))
#define tex_conv_last_4_off_2_2(input_tex_name) (input_tex_name##_texOff(vec2(1, 1)))
vec4 hook() {
    vec4 result = vec4(0.0);
    result += mat4(0.300000011920929, 0.05000000074505806, 0, 0, 0.1000000014901161, 0, 0.05000000074505806, 0, 0, 0, 0, 0.05000000074505806, 0, 0, 0, 0) * tex_conv_last_4_off_0_0(prelu_main_3);
    result += mat4(0.1000000014901161, 0, 0, 0, -0.2000000029802322, 0, 0, 0, 0.1000000014901161, 0, 0, 0, 0, 0, 0, 0) * tex_conv_last_4_off_0_1(prelu_main_3);
    result += mat4(0, 0, 0, 0, 0.1000000014901161, 0, 0, 0, 0.2000000029802322, 0, 0, 0, 0, 0, 0, 0) * tex_conv_last_4_off_0_2(prelu_main_3);
    result += mat4(0.1000000014901161, 0, 0, 0, -0.2000000029802322, 0, 0, 0, 0.1000000014901161, 0, 0, 0, 0, 0, 0, 0) * tex_conv_last_4_off_1_0(prelu_main_3);
    result += mat4(0.4000000059604645, 0.05000000074505806, 0, 0, 0.5, 0, 0.05000000074505806, 0, 0.300000011920929, 0, 0, 0.05000000074505806, 0, 0, 0, 0) * tex_conv_last_4_off_1_1(prelu_main_3);
    result += mat4(0.1000000014901161, 0, 0, 0, -0.2000000029802322, 0, 0, 0, 0.1000000014901161, 0, 0, 0, 0, 0, 0, 0) * tex_conv_last_4_off_1_2(prelu_main_3);
    result += mat4(0, 0, 0, 0, 0.1000000014901161, 0, 0, 0, 0.2000000029802322, 0, 0, 0, 0, 0, 0, 0) * tex_conv_last_4_off_2_0(prelu_main_3);
    result += mat4(0.1000000014901161, 0, 0, 0, -0.2000000029802322, 0, 0, 0, 0.1000000014901161, 0, 0, 0, 0, 0, 0, 0) * tex_conv_last_4_off_2_1(prelu_main_3);
    result += mat4(0.300000011920929, 0.05000000074505806, 0, 0, 0.1000000014901161, 0, 0.05000000074505806, 0, 0, 0, 0, 0.05000000074505806, 0, 0, 0, 0) * tex_conv_last_4_off_2_2(prelu_main_3);
    result += vec4(0.09000000357627869, 0.1000000014901161, 0.1099999994039536, 0);
    return result;
}

//!DESC Compact_SIMPLE_COMPACT_MODEL_PixelShuffle-DirectOutput
//!HOOK MAIN
//!BIND MAIN
//!BIND conv_last_4
//!SAVE MAIN
//!WIDTH conv_last_4.w
//!HEIGHT conv_last_4.h
vec4 hook() {
    vec4 base = MAIN_tex(MAIN_pos);
    vec3 sr_part = vec3(conv_last_4_tex(conv_last_4_pos).rgb);
    return vec4(base.rgb + sr_part, 1.0);
}

