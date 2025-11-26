<cfml>
    <model type="neomind">
        <vocab_size>50000</vocab_size>
        <hidden_size>4096</hidden_size>
        <intermediate_size>12288</intermediate_size>
        <num_hidden_layers>6</num_hidden_layers>
        <num_attention_heads>64</num_attention_heads>
        <num_key_value_heads>64</num_key_value_heads>

        <resid_pdrop>0.1</resid_pdrop>
        <embd_pdrop>0.1</embd_pdrop>
        <attention_dropout>0.1</attention_dropout>

        <hidden_act>gelu</hidden_act>

        <max_position_embeddings>65536</max_position_embeddings>
        <original_max_position_embeddings>8192</original_max_position_embeddings>

        <use_cache>true</use_cache>
        <tie_word_embeddings>false</tie_word_embeddings>

        <rope_theta>10000.0</rope_theta>
        <rope_scaling>
            <type>longrope</type>
            <short_factor>[1.0, 1.0, 1.0, ...]</short_factor>
            <long_factor>[1.0, 1.0, 1.0, ...]</long_factor>
        </rope_scaling>

        <bos_token_id>1</bos_token_id>
        <eos_token_id>49999</eos_token_id>
        <pad_token_id>0</pad_token_id>

        <sliding_window>2048</sliding_window>
    </model>
</cfml>
