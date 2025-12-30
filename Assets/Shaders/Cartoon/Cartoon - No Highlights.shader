Shader "Unlit/CartoonWithCameraOutline"
{
    Properties
    {
        _MainColor("Color", Color) = (0,0.3,1,1)
        _LightStepFalloff("Light Step Falloff", Range(0, 1)) = 0.1
        _AmbientLight("Ambient Light", Color) = (0.0,0.015,0.15, 1)
        _Gloss("Gloss", float) = 15
        _GlossStepFalloff("Gloss Step Falloff", Range(0, 1)) = 0.1

        _OutlineColor("Outline Color", Color) = (1,0,0,1)
        _Width("Outline Width", Range(0, 1)) = 0.1

        _Frequency("Frequency", float) = 20
        _Speed("Speed", float) = 0.5
        _Amplitude("Amplitude", float) = 0.1
        _Axis("Axis", Vector) = (0.1, 1, 0.1, 0)
        
    }

    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "RenderType"="Opaque" }

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

        CBUFFER_START(UnityPerMaterial)
            float4 _MainColor;
            float4 _AmbientLight;
            float  _Gloss;
            float  _LightStepFalloff;
            float  _GlossStepFalloff;

            float4 _OutlineColor;
            float  _Width;

            float  _Frequency;
            float  _Speed;
            float  _Amplitude;
            float4 _Axis;
        CBUFFER_END


        float3 DeformOS(float3 posOS)
        {
            float t = _Speed * _Time.y * 200.0;
            float3 s = sin(t + posOS * _Frequency) * _Amplitude;
            float3 axis = _Axis.xyz;
            return posOS + (s * axis);
        }
        ENDHLSL

        Pass
        {
            Name "Toon"
            Tags { "LightMode"="UniversalForward" }
            Cull Back
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 normalWS   : TEXCOORD0;
                float3 positionWS : TEXCOORD1;
            };

            Varyings vert (Attributes v)
            {
                Varyings o;
                float3 ogPosOS = v.positionOS.xyz;  
                float3 posOS = DeformOS(v.positionOS.xyz);

                float3 posWS = TransformObjectToWorld(ogPosOS);
                float3 nWS   = TransformObjectToWorldNormal(v.normalOS);

                o.positionWS = posWS;
                o.normalWS   = nWS;
                o.positionCS = TransformWorldToHClip(posWS);

                return o;
            }

            half4 frag (Varyings i) : SV_Target
            {
                float3 normalWS = i.normalWS;

                Light mainLight = GetMainLight();
                float3 lightDir = normalize(mainLight.direction);

                // Diffuse (toon)
                float ndl = saturate(dot(normalWS, lightDir));
                float toonDiffuse = step(_LightStepFalloff, ndl);

                // View + reflection
                float3 viewDir = normalize(_WorldSpaceCameraPos - i.positionWS);
                float3 reflectDir = reflect(-viewDir, normalWS);

                // Specular (toon)
                float spec = saturate(dot(reflectDir, lightDir));
                spec = pow(spec, _Gloss);
                float toonSpecular = step(_GlossStepFalloff, spec);

                // Combine
                float3 lit = _AmbientLight.rgb + mainLight.color.rgb * toonDiffuse;
                lit *= _MainColor.rgb;
                lit += mainLight.color.rgb * toonSpecular;

                return half4(lit, 1.0);
            }
            ENDHLSL
        }

Pass
        {
            Name "Outline"
            Cull Front

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
            };

            Varyings vert (Attributes v)
            {
                Varyings o;

                float GROUND_OUTLINE_OFFSET = 0.01;

                float3 posOS = DeformOS(v.positionOS.xyz);

                float3 posWS = TransformObjectToWorld(v.positionOS.xyz);
                float3 nWS   = normalize(TransformObjectToWorldNormal(v.normalOS));

                float distToCam = distance(posWS, _WorldSpaceCameraPos);

                float widthWS = _Width * GROUND_OUTLINE_OFFSET * distToCam;
                posWS += nWS * widthWS;

                o.positionCS = TransformWorldToHClip(posWS);
                return o;
            }

            half4 frag (Varyings i) : SV_Target
            {
                return _OutlineColor;
            }
            ENDHLSL
        }
    }
}
