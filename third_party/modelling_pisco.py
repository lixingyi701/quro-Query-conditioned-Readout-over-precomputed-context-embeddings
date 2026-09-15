import warnings
import os
import torch
import gc

from torch import nn
from jinja2.exceptions import TemplateError
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download


def get_first_layers_model(base_model_name: str, n_layers: int, attn_implementation: str = 'flash_attention_2'):
    """
    Builds a model comprising only the n_layers first layer of the base_model_name
    (it keeps the embedding and head layers)
    """
    full_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Create a new config for a model with fewer layers (e.g., 3 layers)
    custom_config = AutoConfig.from_pretrained(base_model_name)
    custom_config.num_hidden_layers = n_layers 
    first_layers_model = AutoModelForCausalLM.from_config(config=custom_config, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16)       
    
    # Load the state dict of the full model
    full_state_dict = full_model.state_dict()
    custom_state_dict = first_layers_model.state_dict()
    kept_state_dict = {k:v for k,v in full_state_dict.items() if k in custom_state_dict}
    
    first_layers_model.load_state_dict(kept_state_dict, strict=True)
    
    del full_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return first_layers_model


def get_every_n_layer_model(base_model_name: str, every_n_layer: int, attn_implementation: str = 'flash_attention_2'):
    """
    Builds a model comprising 1 every every_n_layer layer of the base_model_name
    (it keeps the embedding and head layers)
    """
    full_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    n_kept_layers = full_model.config.num_hidden_layers // every_n_layer
    
    print(f'New model with 1/{every_n_layer} from {base_model_name} will have {n_kept_layers} layers')
    
    custom_config = AutoConfig.from_pretrained(base_model_name)
    custom_config.num_hidden_layers = n_kept_layers 
    custom_model = AutoModelForCausalLM.from_config(config=custom_config, 
                                                    attn_implementation=attn_implementation, 
                                                    torch_dtype=torch.bfloat16)       
    full_state_dict = full_model.state_dict()
    custom_state_dict = custom_model.state_dict()
    
    # Filter out every Nth layer and rename to form a new state dict
    kept_state_dict = {}
    for key, value in full_state_dict.items():
        if ".layers." in key:
            # Extract layer index
            layer_idx = int(key.split(".layers.")[1].split(".")[0])
            # Check if it's an Nth layer
            if layer_idx % every_n_layer == 0:
                # Adjust layer index for the smaller model
                new_layer_idx = layer_idx // every_n_layer
                # print('replacing', f".layers.{layer_idx}.", f".layers.{new_layer_idx}.")
                new_key = key.replace(f".layers.{layer_idx}.", f".layers.{new_layer_idx}.")
                if new_key in custom_state_dict:
                    kept_state_dict[new_key] = value
        else:
            # Keep non-layer-specific parameters
            if key in custom_state_dict:
                kept_state_dict[key] = value

    # Load the filtered state dict into the custom model
    custom_model.load_state_dict(kept_state_dict, strict=True)
    
    del full_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return custom_model
                
    
class MistralTrimmed(torch.nn.Module):
    """
    Trimmed version of base models for faster compression
    NB: the name 'MistralTrimmed' suggests it just works with mistral but NO in fact most LLMs are supported !
    """
    def __init__(self, 
                 n_layers: int = 15,
                 every_n_layer: int = None,
                 rms_norm: bool = False,
                 base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 attn_implementation: str = 'flash_attention_2'):
        """
        you can either specify
        - n_layers to some number: we take the n_layers first layers of the base model.
        - every_n_layer to some number: in that case we take 1/N layer of the base model
        The base_model_name is the name of the model from which this model is built.
        """
        assert (n_layers is None) ^ (every_n_layer is None), 'Cannot specify both n_layers and every_n_layer for MistralTrimmed'        
        super().__init__()
        
        self.n_layers = n_layers
        self.every_n_layer = every_n_layer
        self.base_model_name = base_model_name
        
        if n_layers is not None:
            self.custom_model = get_first_layers_model(self.base_model_name, 
                                                       n_layers, 
                                                       attn_implementation=attn_implementation)
        
        else:
            self.custom_model = get_every_n_layer_model(self.base_model_name, 
                                                        every_n_layer, 
                                                        attn_implementation=attn_implementation)
            
        self.custom_model = self.custom_model.bfloat16()
        self.custom_model.cuda()

        if rms_norm:
            print('Compressor keeps its original rms norm')
        else:
            print('De-activating RMS norm in compressor')
            # We deactivate the norm: we don't need it here since we want to manipulate stuff within embed space
            # see https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/models/mistral/modeling_mistral.py#L699
            self.custom_model.model.norm = nn.Identity()
        
        # Piping useful methods:
        self.add_adapter = self.custom_model.add_adapter
        self.set_adapter = self.custom_model.set_adapter
        self.load_adapter = self.custom_model.load_adapter
        self.num_parameters = self.custom_model.num_parameters
        self.resize_token_embeddings = self.custom_model.resize_token_embeddings
        self.get_input_embeddings = self.custom_model.get_input_embeddings
        self.get_adapter_state_dict = self.custom_model.get_adapter_state_dict
        
        # self.custom_model.gradient_checkpointing_enable()
        
        # del self.custom_model.lm_head # THIS FAILS since some models have tie_embeddings=True !
        # gc.collect()
        # torch.cuda.empty_cache()    

    def forward(self, input_ids, attention_mask=None):
        return self.custom_model.model(input_ids, attention_mask, output_hidden_states=True) # we call the .model attribute of the causal LM to avoid the cost of the LM head ! nice huh ?
    
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        return self.forward(input_ids, attention_mask)


class AbstractCompressor(nn.Module):
    def __init__(self, compr_model_name: str, compr_rate: int, decoder_hidden_size: int):
        super().__init__()
        self.compr_model_name = compr_model_name
        self.compr_rate = compr_rate
        self.decoder_hidden_size = decoder_hidden_size
        
    def forward(self, input_ids, attention_mask, generation_top_k):
        """
        input_ids of shape (batch_size, top_k, seq_length)
        attention_mask of shape (batch_size, top_k, seq_length)
        generation_top_k: the number of docs
        """
        raise NotImplementedError
    
    def save_pretrained(self, save_directory):
        raise NotImplementedError

    def load_pretrained(self, load_directory):
        raise NotImplementedError


class BertCompressor(AbstractCompressor):
    def __init__(self, 
                 compr_model_name: str,
                 compr_rate: int, 
                 decoder_hidden_size: int, 
                 mlp_hidden_dim: int = 8192, 
                 use_mlp: bool = True,
                 doc_max_length : int = 128,
                 **kwargs):
        # TODO use the device_map
        super().__init__(compr_model_name=compr_model_name, compr_rate=compr_rate, decoder_hidden_size=decoder_hidden_size)
        if compr_model_name == 'mistral_trimmed':
            assert 'compr_n_layers' in kwargs
            self.model = MistralTrimmed(n_layers=kwargs['compr_n_layers'], 
                                        every_n_layer=kwargs['compr_every_n_layer'], 
                                        rms_norm=kwargs['compr_rms_norm'],
                                        base_model_name=kwargs['compr_base_model_name'],
                                        attn_implementation=kwargs['attn_implementation'])
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.base_model_name)
            self.hidden_size = self.model.custom_model.config.hidden_size
        else:
            self.model = AutoModel.from_pretrained(compr_model_name, torch_dtype=torch.bfloat16, device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained(compr_model_name, use_fast=True)
            self.tokenizer.padding_side = "left"
            self.hidden_size = self.model.config.hidden_size
            
        print('Base compressor nb parameters', self.model.num_parameters())

        self.mlp_hidden_dim = mlp_hidden_dim
        self.use_mlp = use_mlp
        self.doc_max_length = doc_max_length
        
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.mlp_hidden_dim), 
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_dim, decoder_hidden_size)
            ).bfloat16() 
            self.mlp.cuda()
        
        self.n_emb = self.doc_max_length // self.compr_rate
        
        mem_tokens = ['<MEM' + str(i) + '>' for i in range(self.n_emb)]
        self.tokenizer.add_special_tokens({'additional_special_tokens': mem_tokens}) 
        self.tokenizer.mem_tokens = mem_tokens
        self.tokenizer.mem_token_ids = [self.tokenizer.convert_tokens_to_ids(elt) for elt in self.tokenizer.mem_tokens]
        self.tokenizer.mem_token_ids_pt = torch.LongTensor(self.tokenizer.mem_token_ids)
        self.model.resize_token_embeddings(len(self.tokenizer))
            
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
            
        if not use_mlp:
            assert decoder_hidden_size == self.hidden_size, f'Mlp mandatory is hidden sizes not equal: {decoder_hidden_size} vs {self.hidden_size}'
            
        self.lora = False
        self.lora_name = 'compr_adapter'
        
    def prepare_mem_tokens_optimization(self):
        assert self.lora, 'should only be called with lora.'
        self.model.get_input_embeddings().weight.requires_grad = True
        # Applying a hook zero-ing the gradients except for the mem token:
        def hook(grad):
            mask = torch.zeros_like(grad)
            mask[self.tokenizer.mem_token_ids] = 1.0
            return grad * mask
        self.model.get_input_embeddings().weight.register_hook(hook)            
        
    def set_lora(self, peft_config):
        self.model.add_adapter(peft_config, self.lora_name)
        self.model.set_adapter(self.lora_name)
        self.lora = True
        self.prepare_mem_tokens_optimization()

    def forward(self, input_ids, attention_mask):
        assert input_ids.size() == attention_mask.size()
        assert len(input_ids.size()) == 2
        
        batch_size_times_top_k = input_ids.size(0)
        
        last_hidden_states = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask, 
                                        output_hidden_states=True).hidden_states[-1]
        
        # Getting the hidden states at the mem token positions, as for regular cocom:
        mask = torch.isin(input_ids, self.tokenizer.mem_token_ids_pt.to(input_ids.device))
        selected_n_tokens = last_hidden_states[mask].reshape(last_hidden_states.size(0), -1, last_hidden_states.size(-1))            
        
        assert selected_n_tokens.size() == (batch_size_times_top_k, self.n_emb, self.hidden_size), f"{selected_n_tokens.size()} vs {(batch_size_times_top_k, self.n_emb, self.hidden_size)}"
        
        if self.use_mlp:
            selected_n_tokens = self.mlp(selected_n_tokens) # now of shape (batch_size, top_k, decoder_hidden_size)
            
        assert selected_n_tokens.size() == (batch_size_times_top_k, self.n_emb, self.decoder_hidden_size), f"{selected_n_tokens.size()} vs {(batch_size_times_top_k, self.n_emb, self.decoder_hidden_size)}"
        
        return selected_n_tokens
    
    def get_lora_path_from_directory(self, directory):
        return os.path.join(directory, 'compressor_adapters.pth')
    
    def get_compressor_path_from_directory(self, directory):
        return os.path.join(directory, 'compressor.pth')
    
    def get_mlp_path_from_directory(self, directory):
        return os.path.join(directory, 'mlp.pth')
    
    def get_first_layer_path_from_directory(self, directory):
        return os.path.join(directory, 'first_layer.pth')
    
    def get_first_layer_state_dict(self) -> dict:
        out = {}
        for k, v in self.model.named_parameters():
            if 'embed_tokens.weight' in k:
                out[k] = v.cpu()
                
        assert len(out) == 1, len(out) # We should get exactly one layer here
        return out
    
    def save_pretrained(self, save_directory):
        """
        Here we just save mlp state_dict and model state_dict
        Config is handled in cocom model.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save MLP weights
        if self.use_mlp:
            mlp_path = self.get_mlp_path_from_directory(directory=save_directory)
            torch.save(self.mlp.state_dict(), mlp_path)
        
        # Saving the model
        if not self.lora: # full training: save the full dict:
            model_path = self.get_compressor_path_from_directory(directory=save_directory)
            torch.save(self.model.state_dict(), model_path)
        else: # lora training of the compressor
            # We save the first layer:
            first_layer_state_dict = self.get_first_layer_state_dict()
            torch.save(first_layer_state_dict, self.get_first_layer_path_from_directory(directory=save_directory))
            
            # We save the adapters:
            adapter_state_dict = {k: v.cpu() for k, v in self.model.get_adapter_state_dict(self.lora_name).items()}
            torch.save(adapter_state_dict, self.get_lora_path_from_directory(directory=save_directory))
            
    def load_adapter(self, load_directory, peft_config):    
        assert peft_config is not None
        map_location = torch.device("cpu") if not torch.cuda.is_available else None
        adapter_state_dict = torch.load(self.get_lora_path_from_directory(directory=load_directory), map_location=map_location, weights_only=True)
        print('loading compr adapter onto compressor model from', self.get_lora_path_from_directory(directory=load_directory))
        self.model.load_adapter(peft_config=peft_config, adapter_name=self.lora_name, adapter_state_dict=adapter_state_dict)
        self.lora = True
        self.prepare_mem_tokens_optimization()
        
    def load_first_layer(self, load_directory):
        map_location = torch.device("cpu") if not torch.cuda.is_available else None
        first_layer_state_dict = torch.load(self.get_first_layer_path_from_directory(load_directory), map_location=map_location, weights_only=True)
        assert len(first_layer_state_dict.keys()) == 1
        self.model.load_state_dict(first_layer_state_dict, strict=False)

    def load_pretrained(self, load_directory, lora: bool = False, peft_config=None):
        """
        Loading the state dicts.
        :lora: if True then the compressor was trained using lora: we just need to load the adapters
        if False, the compressor was fully trained: we load it fully.
        """
        if self.use_mlp:
            mlp_path = self.get_mlp_path_from_directory(directory=load_directory)
            self.mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
        
        if lora:
            self.load_first_layer(load_directory)
            self.load_adapter(load_directory, peft_config)
            
        else:
            model_path = self.get_compressor_path_from_directory(directory=load_directory)
            self.model.load_state_dict(torch.load(model_path, weights_only=True))  
            
    def prepare_inputs(self, texts, max_length, q_texts=None):
        if q_texts is not None: # Query-dependent here:
            assert len(texts) == len(q_texts), f"{len(texts)} == {len(q_texts)}"
            if self.compr_model_name == 'mistral_trimmed':
                # No special token, just formulating:
                texts_to_encode = [ '\nQuery:\n' + query + 'Document:\n' + text for text, query in zip(texts, q_texts)]
                inp_enc = self.tokenizer(texts_to_encode, 
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        max_length=max_length + 8, # some margin for query/doc stuff + bos / eos
                                        truncation=True,
                                        add_special_tokens=True)
            else:
                inp_enc = self.tokenizer(q_texts,  # we put the query in first position
                                        texts, 
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        max_length=max_length + 3,
                                        truncation='only_second',
                                        add_special_tokens=True)
        else:
            inp_enc = self.tokenizer(texts, return_tensors='pt', padding='max_length', max_length=max_length + 2, truncation=True, add_special_tokens=True)
            
        inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(inp_enc['input_ids'], 
                                                                                        inp_enc['attention_mask'], 
                                                                                        self.n_emb, 
                                                                                        tokenizer=self.tokenizer)
            
        return inp_enc


def add_memory_tokens_to_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor, n_mem_tokens: int, tokenizer):
    """
    Concatenate the input ids with n_mem_tokens mem_tokens and update the corresponding attention mask
    """
    assert len(tokenizer.mem_tokens) == n_mem_tokens, f"{len(tokenizer.mem_tokens)} VS {n_mem_tokens}"
    mem_tokens = torch.stack([tokenizer.mem_token_ids_pt] * input_ids.size(0), 0)
    assert len(mem_tokens.size()) == 2
    assert len(mem_tokens) == input_ids.size(0)
    assert len(mem_tokens[0]) == n_mem_tokens
    #mem_tokens = torch.full((input_ids.size(0), n_mem_tokens), tokenizer.mem_token_id, dtype=torch.long)
    input_ids = torch.cat([input_ids, mem_tokens], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(input_ids.size(0), n_mem_tokens)], dim=1)
    return input_ids, attention_mask


class COCOMConfig(PretrainedConfig):

    model_type = "COCOM"
    def __init__(self,
                decoder_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                doc_max_length: int = 128,
                quantization: str = 'no',
                sep: bool = False,
                compr_model_name: str = "google-bert/bert-base-uncased",
                compr_rate: int = 64,
                compr_n_layers: int = None, # only for surgical mistral compressor
                compr_every_n_layer: int = None,
                compr_base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                compr_rms_norm: bool = False, # only for surgical mistral compressor: if true, rms norm applied on h-s
                compr_mlp_hidden_dim: int = 8096,
                compr_use_mlp: bool = True, 
                lora: bool = False, # lora on decoder (and decoder as compr)
                lora_compressor: bool = False, # lora only on the compressor if it exists
                training_form: str = "both",
                lora_r: int = 16,
                lora_r_compressor: int = None,
                load_adapters: bool = True,
                kbtc_training: bool = False,
                optimize_mem_tokens: bool = False,
                different_mem_tokens: bool = False,
                attn_implementation: str = 'flash_attention_2',
                device_map = None,
                **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name # model name of decoder
        self.doc_max_length = doc_max_length # the maximum length of document that can be used by this model (it is used to compute number of mem tokens !)
        self.quantization = quantization # quantization, could be no, int4, int8
        self.sep = sep # boolean type, whether to use sep token
        
        self.compr_model_name = compr_model_name # model name of compressor
        self.compr_rate = compr_rate # compression rate
        self.compr_use_mlp = compr_use_mlp
        self.compr_mlp_hidden_dim = compr_mlp_hidden_dim
        self.compr_n_layers = compr_n_layers
        self.compr_every_n_layer = compr_every_n_layer
        self.compr_base_model_name = compr_base_model_name
        self.compr_rms_norm = compr_rms_norm
        
        self.lora = lora # boolean type, whether to use lora trsining
        self.lora_compressor = lora_compressor
        self.training_form = training_form # training form, could be compressor: training only comprssor; both: training both
        # Or both_separately: training both with separate adapters
        self.lora_r = lora_r # lora_r for lora training, we use 16 throughout the experiment.
        self.lora_r_compressor = lora_r_compressor or lora_r # defaulting to same lora as decoder.
        self.load_adapters = load_adapters # used to load pretrained model: we first load without adapters, and then load them from file.
        self.optimize_mem_tokens = optimize_mem_tokens
        self.different_mem_tokens = different_mem_tokens
        
        self.kbtc_training = kbtc_training
        
        self.device_map = device_map
        
        self.attn_implementation = attn_implementation
        
        if training_form == 'compressor':
            assert compr_model_name is not None and not self.lora
            
        
class COCOM(PreTrainedModel):
    config_class = COCOMConfig
    def __init__(self, cfg):
        super().__init__(cfg)
        self.decoder_model_name = cfg.decoder_model_name
        self.decoder = self.create_decoder(cfg)
        
        self.doc_max_length = cfg.doc_max_length
        
        print('Base decoder nb parameters', self.decoder.num_parameters())

        self.compr_model_name = cfg.compr_model_name
        self.training_form = cfg.training_form
        self.lora = cfg.lora
        self.adapter_keys = []

        self.compr = None
        # when compr_model_name is not set, then means using a decoder-based compressor, otherwise a bert based compressor
        if cfg.compr_model_name is not None:
            # case bert based compressor
            print('Instantiating compressor ', cfg.compr_model_name)
            self.compr = BertCompressor(cfg.compr_model_name, 
                                        cfg.compr_rate, 
                                        doc_max_length=self.doc_max_length,
                                        decoder_hidden_size=self.decoder.config.hidden_size,
                                        mlp_hidden_dim=cfg.compr_mlp_hidden_dim,
                                        compr_n_layers=cfg.compr_n_layers,
                                        compr_every_n_layer=cfg.compr_every_n_layer,
                                        compr_base_model_name=cfg.compr_base_model_name,
                                        compr_rms_norm=cfg.compr_rms_norm,
                                        use_mlp=cfg.compr_use_mlp,
                                        attn_implementation=cfg.attn_implementation)

        # set lora adaptors on decoder model
        if cfg.lora:
            peft_config = self.get_peft_config(lora_r=cfg.lora_r)

            if cfg.load_adapters:
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.decoder.set_adapter('decoder_adapter') # active adapter by default
                self.adapter_keys.append('decoder_adapter')

            # Create separate adapters (if not BERT compressor and training_form == 'both_separately')
            if self.training_form == 'both_separately' and self.compr is None:
                if cfg.load_adapters:
                    self.decoder.add_adapter(peft_config, 'encoder_adapter')
                    self.adapter_keys.append('encoder_adapter')
                    
        # set lora adapters on compressor model:
        if cfg.lora_compressor and self.compr is not None and cfg.load_adapters:
            peft_config = self.get_peft_config(lora_r=cfg.lora_r_compressor)
            self.compr.set_lora(peft_config)
                    
        self.decoder_tokenizer = COCOM.create_decoder_tokenizer(cfg)

        # resize the tokenizer embedding
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self.decoder.generation_config.top_p = None
        self.decoder.generation_config.temperature = None
        self.decoder.generation_config.pad_token_id = self.decoder_tokenizer.pad_token_id
        
        # self.decoder.gradient_checkpointing_enable()
        # if self.compr is not None:
        #     self.compr.gradient_checkpointing_enable()

        # other settings
        self.generation_top_k = 1
        self.sep = cfg.sep
        self.compr_rate = cfg.compr_rate
        self.local_rank = os.getenv('LOCAL_RANK', '0')
        
        self.n_mem_tokens = self.doc_max_length // self.compr_rate # crucial!


        if self.lora:
            for adapter_key in self.adapter_keys:
                self.decoder.set_adapter(adapter_key)
                print(f'Adapter {adapter_key} trainable parameters: {self.num_parameters(only_trainable=True)}')
                
            #  We need to activate all adapters so that they are both trained...
            self.set_all_adapters()
        else:
            print(f'Total trainable parameters: {self.num_parameters(only_trainable=True)}')
            
        if self.compr is not None:
            print(f'Compressor number of parameters: {self.compr.model.num_parameters(only_trainable=True)}')
            
        self.prepare_mem_tokens_optimization()
            
    def prepare_mem_tokens_optimization(self):
        if self.config.optimize_mem_tokens:
            if self.compr is None:
                # Enforcing gradients for input embeddings (even if lora)
                self.decoder.get_input_embeddings().weight.requires_grad = True
                # Applying a hook zero-ing the gradients except for the mem token:
                def hook(grad):
                    mask = torch.zeros_like(grad)
                    mask[self.decoder_tokenizer.mem_token_ids] = 1.0
                    return grad * mask
                self.decoder.get_input_embeddings().weight.register_hook(hook)
                
    def set_all_adapters(self):
        if len(self.adapter_keys) > 0:
            self.decoder.set_adapter(self.adapter_keys)
            
    @staticmethod
    def create_decoder_tokenizer(cfg: COCOMConfig):
        decoder_tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name, use_fast=True, padding_side='left')

        # define special tokens
        n_mem_tokens = cfg.doc_max_length // cfg.compr_rate
        if cfg.different_mem_tokens:
            # estimation fo the number of memory tokens needed:
            mem_tokens = ['<MEM' + str(i) + '>' for i in range(n_mem_tokens)]
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': mem_tokens + ['<AE>', '<ENC>', '<SEP>']}) 
            decoder_tokenizer.mem_tokens = mem_tokens
        else:
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>', '<AE>', '<ENC>', '<SEP>']})
            decoder_tokenizer.mem_tokens = ['<MEM>'] * n_mem_tokens
        
        decoder_tokenizer.mem_token_ids = [decoder_tokenizer.convert_tokens_to_ids(elt) for elt in decoder_tokenizer.mem_tokens]
        decoder_tokenizer.mem_token_ids_pt = torch.LongTensor(decoder_tokenizer.mem_token_ids) # required later on for operations on tensors
        
        decoder_tokenizer.ae_token = '<AE>' # token for autoencoding on decoder side
        decoder_tokenizer.ae_token_id = decoder_tokenizer.convert_tokens_to_ids('<AE>')
        decoder_tokenizer.enc_token = '<ENC>' # token for autoencoding on compressor side
        decoder_tokenizer.sep_token = '<SEP>' # sep token between document
        decoder_tokenizer.sep_token_id = decoder_tokenizer.convert_tokens_to_ids('<SEP>')

        # If kbtc training, we add another one yet
        if cfg.kbtc_training:
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<KBTC>']})
            decoder_tokenizer.kbtc_token = '<KBTC>'
            decoder_tokenizer.kbtc_token_id = decoder_tokenizer.convert_tokens_to_ids('<KBTC>')

        # if pad token exists then use pad token, othrwise bos token
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token_id = decoder_tokenizer.bos_token_id

        return decoder_tokenizer

    def get_peft_config(self, lora_r: int) -> LoraConfig:
        """
        Builds the peft config
        """
        return LoraConfig(task_type="CAUSAL_LM", r=lora_r, lora_alpha=2*lora_r, target_modules='all-linear', lora_dropout=0.1)

    def create_decoder(self, cfg):
        """
        Loads the base decoder.
        """
        if torch.cuda.is_available():
            if cfg.quantization == "no":
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=self.config.attn_implementation,
                    # low_cpu_mem_usage = True,
                    device_map=cfg.device_map
                    )
            elif cfg.quantization == "int4":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='bfloat16',
                    # low_cpu_mem_usage = True,
                )
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name,
                    quantization_config=quant_config,
                    attn_implementation=self.config.attn_implementation,
                    torch_dtype=torch.bfloat16,
                    resume_download=True,
                    # low_cpu_mem_usage = True,
                    trust_remote_code=True,
                    device_map=cfg.device_map
                )
            elif cfg.quantization == "int8":
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_4bit_compute_dtype='bfloat16',
                    # low_cpu_mem_usage = True,
                )
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name,
                    quantization_config=quant_config,
                    attn_implementation=self.config.attn_implementation,
                    torch_dtype=torch.bfloat16,
                    resume_download=True,
                    # low_cpu_mem_usage = True,
                    trust_remote_code=True,
                    device_map=cfg.device_map
                )
            else:
                raise NotImplementedError()
        else:
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                # low_cpu_mem_usage = True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )
            
    def compress(self, enc_input_ids, enc_attention_mask):
        if self.compr:
            return self.compr(enc_input_ids, enc_attention_mask)
        else:
            return self.compr_decoder(enc_input_ids, enc_attention_mask)

    def replace_emb(self, compressed_embs, dec_input_ids):
        """
        Compression logic (either with decoder or with dedicated compressor)
        """
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k)            
        input_embeds = self.replace_embeddings(compressed_embs, dec_input_ids, indices)
        return input_embeds

    def compr_decoder(self, input_ids, attention_mask):
        """
        Compression using the decoder
        """
        assert input_ids.size() == attention_mask.size(), f"{input_ids.size()} vs {attention_mask.size()}"
        
        # Switch adapter if we are training two different ones:
        if 'encoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('encoder_adapter')

        emb = self.decoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True).hidden_states[-1]
        mask = torch.isin(input_ids, self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device))
        return emb[mask].reshape(emb.size(0), -1, emb.size(-1))
    
    def prepare_encoder_inputs_to_decoder(self, texts, max_length, q_texts=None):
        if q_texts is not None:
            texts_to_encode = [self.decoder_tokenizer.enc_token + self.decoder_tokenizer.bos_token + '\nQuery:\n' + query + 'Document:\n' + text + self.decoder_tokenizer.eos_token 
                               for text, query in zip(texts, q_texts)]
            inp_enc = self.decoder_tokenizer(texts_to_encode, return_tensors='pt', padding='max_length', max_length=max_length + 8, truncation=True, add_special_tokens=False)
        else:
            inp_enc = [self.decoder_tokenizer.enc_token + self.decoder_tokenizer.bos_token + text + self.decoder_tokenizer.eos_token for text in texts]
            inp_enc = self.decoder_tokenizer(inp_enc, return_tensors='pt', padding="max_length", max_length=max_length+3, truncation=True, add_special_tokens=False)
            
        num_mem_tokens = self.doc_max_length // self.compr_rate
        assert num_mem_tokens == len(self.decoder_tokenizer.mem_tokens)
        inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(inp_enc['input_ids'], 
                                                                                        inp_enc['attention_mask'], 
                                                                                        num_mem_tokens, 
                                                                                        tokenizer=self.decoder_tokenizer)
        
        return inp_enc
    
    def prepare_encoder_inputs(self, texts: list[str], max_length: int, q_texts: list[str] = None):
        """
        Create the inputs to the encoder, for compression.
        """
        if q_texts is not None:
            assert len(texts) == len(q_texts), f"{len(texts)} == {len(q_texts)}"

        # Case where the encoder is the decoder with adapter:
        if self.compr is None:
            return self.prepare_encoder_inputs_to_decoder(texts, max_length, q_texts)
        
        # Case where the encoder is a separate network:
        else:
            return self.compr.prepare_inputs(texts, max_length, q_texts)

    def replace_embeddings(self, compressed_embs, dec_input_ids, indices):
        """
        Replace memory tokens in the decoder input to with the compressed embeddings
        """
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)
        if self.sep:
            slot_len = num_embs + 1
        else:
            slot_len = num_embs
        # get first mem_token indices
        first_mem_token_indices = torch.argmax((dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]).int(), dim=1)
        batch_size = inputs_embeds.size(0)
        # for each example in batch, replace them with compressed embeddings
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j-indices[i]) * slot_len
                assert inputs_embeds[i, start_idx:start_idx + num_embs, :].size() == compressed_embs[j].size(), \
                    f"{inputs_embeds[i, start_idx:start_idx + num_embs, :].size()} VS {compressed_embs[j].size()}"
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        return inputs_embeds

    def forward(self,
                enc_input_ids: torch.LongTensor = None,
                enc_attention_mask: torch.LongTensor = None,
                dec_input_ids: torch.LongTensor = None,
                dec_attention_mask: torch.LongTensor = None,
                labels: torch.LongTensor = None):
        """
        enc_input_ids: stores the contexts, should be flattened from all queries before input, can be of shape:
            - (batch_size*generation_top_k, enc_token_length)
            - (batch_size, generation_top_k, enc_token_length)
        enc_attention_mask: attention mask of enc_input_ids, same shape as enc_input_ids
        dec_input_ids: stores the prompts (including mem tokens), dimention (batch_size, dec_token_length)
        dec_attention_mask: attention mask of dec_input_ids
        """ 
        assert enc_input_ids.size() == enc_attention_mask.size(), f"{enc_input_ids.size()} vs {enc_attention_mask.size()}"
        
        if len(enc_input_ids.size()) == 3: # likely from bergen: we just flatten all of this to perform encoding in one batch
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
        
        # Here, we should have top_k times more elements in enc_input_ids than in dec_input_ids
        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k, \
            f"{enc_input_ids.size(0)} VS {dec_input_ids.size(0)} with generation_top_k={self.generation_top_k}"
            
        # Perform compression with gradient tracking
        compressed_embs = self.compress(enc_input_ids, enc_attention_mask)
        inputs_embeds = self.replace_emb(compressed_embs, dec_input_ids)

        # if training_form is compressor, then detach the inputs_embeds, to make gradient not count in decoder
        if (self.training_form == "compressor") and (self.compr is None):
            inputs_embeds  = inputs_embeds.detach()

        # decoding
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=dec_attention_mask, labels=labels)

        # At end of forward, we need to activate all adapters so that they are both trained...
        self.set_all_adapters()

        return {"loss": decoder_outputs.loss, "logits": decoder_outputs.logits}

    def generate(self, model_input, max_new_tokens=128, return_doc_embeddings: bool = False):

        enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask = model_input['enc_input_ids'], model_input['enc_attention_mask'], model_input['dec_input_ids'], model_input['dec_attention_mask']
        
        assert enc_input_ids.size() == enc_attention_mask.size()
        
        if len(enc_input_ids.size()) == 3: # likely from bergen: we just flatten all of this to perform encoding in one batch
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
            
        # Here, we should have top_k times more elements in enc_input_ids than in dec_input_ids
        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k, \
            f"{enc_input_ids.size(0)} VS {dec_input_ids.size(0)} with generation_top_k={self.generation_top_k}"
            
        compressed_embs = self.compress(enc_input_ids.to('cuda'), enc_attention_mask.to('cuda'))
        inputs_embeds = self.replace_emb(compressed_embs, dec_input_ids.to('cuda'))
        
        # Switch adapter if we are training two different ones:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter') 

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=dec_attention_mask.to("cuda"),
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens
            )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        if return_doc_embeddings:
            # Compressed_embds is of shape (batch_size*top_k, n_mem_tokens, hidden_dim)
            # We reshape to batch_size, top_k, n_mem_tokens, hidden_dim
            assert batch_size is not None
            assert top_k is not None
            compressed_embs = compressed_embs.view(batch_size, top_k, compressed_embs.size(1), compressed_embs.size(2))
            return decoded, compressed_embs
        else:
            return decoded

    def get_all_adapters_state_dict(self):
        """
        Return the state dicts of the adapters
        Used for saving so we go to cpu automatically
        """
        return {key: {k:v.cpu() for k, v in self.decoder.get_adapter_state_dict(key).items()} for key in self.adapter_keys}

    def load_adapter_from_state_dict(self, peft_config: LoraConfig, adapter_name: str, adapter_state_dict: dict) -> None:
        """
        Creates an adapter from the state dict (used to load from pretrained)
        """
        # assert adapter_name not in self.adapter_keys, f'Adapter {adapter_name} already exists'
        print(f'loading adapter {adapter_name}')
        self.decoder.load_adapter(peft_config=peft_config, adapter_name=adapter_name, adapter_state_dict=adapter_state_dict)
        self.adapter_keys.append(adapter_name)
        
    def get_decoder_first_and_last_layer_state_dict(self) -> dict:
        """
        Just getting the first and last layers: the only ones which change when adding tokens
        Used to save the model so we automatically move to cpu.
        """
        out = {}
        for k, v in self.decoder.named_parameters():
            if 'lm_head.weight' in k or 'embed_tokens.weight' in k:
                out[k] = v.cpu()
                
        # assert len(out) == 2, len(out) # We should get both the embedding layer and the head layer.
        return out

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save only the LoRA adapters and their configurations.
        """
        if self.lora:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory) 

            # Save the LoRA adapter weights
            torch.save(self.get_all_adapters_state_dict(), os.path.join(save_directory, "adapters.pth"))
            
            # Save the first and last layers of decoder (because of diffs with tokens !)
            torch.save(self.get_decoder_first_and_last_layer_state_dict(), os.path.join(save_directory, "decoder_first_last_layers.pth"))
            
            # Save the bert compressor if it exists
            if self.compr_model_name is not None:
                self.compr.save_pretrained(os.path.join(save_directory, 'compressor'))

            # Save the configuration
            self.config.save_pretrained(save_directory)
        else:
            super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Loading: to take care of checkpoints containing only lora and not base model.
        """
        # Load the configuration
        config = COCOMConfig.from_pretrained(pretrained_model_name_or_path)
        
        config.attn_implementation = kwargs.get('attn_implementation', config.attn_implementation)  
        
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None

        if config.lora:
            # We need to delay the construction of the adapters (otherwise peft complains)
            config.load_adapters = False

            if 'device_map' in kwargs:
                config.device_map = kwargs['device_map']

            # Initialize the model
            model = cls(config)
                    
            # Loading first and last layers (they might have changed due to extra tokens)
            try:
                # If loading from Hugging Face Hub
                first_and_last_layers_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, 
                    filename="decoder_first_last_layers.pth"
                )
            except Exception as e:
                # If loading from a local directory
                first_and_last_layers_path = os.path.join(pretrained_model_name_or_path, "decoder_first_last_layers.pth")

            if os.path.exists(first_and_last_layers_path):
                first_and_last_decoder_state_dict = torch.load(first_and_last_layers_path, map_location=map_location, weights_only=True)
                for key in first_and_last_decoder_state_dict:
                    assert key in model.decoder.state_dict()
                    model.decoder.load_state_dict(first_and_last_decoder_state_dict, strict=False)

            else:
                print('FIRST AND LAST LAYER NOT FOUND (ok for some old models):', first_and_last_layers_path)
                
            peft_config = model.get_peft_config(lora_r=config.lora_r)
            
            # Load the LoRA adapters (if the file exists)
            try:
                # If loading from Hugging Face Hub
                adapters_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, 
                    filename="adapters.pth"
                )
            except Exception as e:
                # If loading from a local directory
                adapters_path = os.path.join(pretrained_model_name_or_path, "adapters.pth")
                
            if os.path.exists(adapters_path):
                adapters_state_dict = torch.load(adapters_path, map_location=map_location, weights_only=True)
            
                for key, val in adapters_state_dict.items():
                    model.load_adapter_from_state_dict(peft_config=peft_config, adapter_name=key, adapter_state_dict=val)
                    
            else:
                warnings.warn(f'I see lora on that PISCO model, but {adapters_path} does not exist, it may be normal \
                        for recent versions of transformers, be aware.')

            # If there is a compressor, it's been built: we just need to load the state dict or the adapters:
            if config.compr_model_name is not None:
                model.compr.load_pretrained(os.path.join(pretrained_model_name_or_path, 'compressor'), 
                                            lora=config.lora_compressor, 
                                            peft_config=model.get_peft_config(lora_r=config.lora_r_compressor))

            model.set_all_adapters()
            model.config.load_adapters = True
            return model

        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
    def generate_from_text(self, questions: list[str], documents: list[list[str]], max_new_tokens: int = 128) -> list[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: list of string
        documents: list of list of strings (they should all be of equal length: the nb of doc for each question)
        """
        self.generation_top_k = len(documents[0])
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])
        flat_documents = sum(documents, [])
        
        model_input = {}
        
        # Creating encoder inputs:
        input_encoder = self.prepare_encoder_inputs(flat_documents, max_length=128)
        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        
        # Creating decoder inputs
        instr = [self.blend_prompt_and_memory_tokens(query=q) for q in questions]
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=2048)
        model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # Generation
        return self.generate(model_input, max_new_tokens=max_new_tokens)
    
    def generate_from_compressed_documents_and_questions(self, questions: list[str], compressed_documents: torch.Tensor, max_new_tokens: int = 128) -> list[str]:
        """
        Generates answers from compressed documents
        questions: list of string
        compressed_documents: torch tensor, its first dimension should be a multiple of len(questions)
        """
        self.generation_top_k = compressed_documents.size(0) // len(questions)
        assert compressed_documents.size(0) % self.generation_top_k == 0, f"{compressed_documents.size(0)} {self.generation_top_k}"
        
        # Creating decoder inputs
        instr = [self.blend_prompt_and_memory_tokens(query=q) for q in questions]
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=2048)
        device = self.decoder.device
        dec_input_ids, dec_attention_mask = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)

        # Creating input decoder embeddings from prompt + compressed documents
        inputs_embeds = self.replace_emb(compressed_documents, dec_input_ids)
        
        # Activating decoder generator:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')
            
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            generation_config=self.generation_config,
            max_new_tokens=max_new_tokens
            )
        
        # de-tokenizing
        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
    def compress_documents(self, documents: list[str]) -> torch.Tensor:
        """
        Compress a list of documents
        """
        input_encoder = self.prepare_encoder_inputs(documents, max_length=128)
        enc_input_ids = input_encoder['input_ids'].to(self.decoder.device)
        attention_mask = input_encoder['attention_mask'].to(self.decoder.device)
        return self.compress(enc_input_ids=enc_input_ids, enc_attention_mask=attention_mask)
    
    def blend_prompt_and_memory_tokens(self, query: str):
        """
        Takes care of blending the prompt with the memory tokens:
        Also returns, if a label is provided, the position of the first token index of the label (for loss comp later on)
        (Used for the HUB version)
        """        
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        
        # proper names for "eval" call, don't remove these lines
        docs = mem_tokens_str * self.generation_top_k
        question = query
        
        prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        prompt_user = f"Background:\n{docs}\n\nQuestion:{question}"
        
        # Prepare the messages with system and user roles
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user.replace(':\ ', ': ')}
        ]

        # Attempt to apply the system role and catch if it's not supported
        try:
            prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        except TemplateError as e:
            # Catch the error related to system role and handle it (e.g. gemma)
            if "System role not supported" in str(e):
                # Remove system role and proceed with only the user role
                messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]
                # Apply template again without system role
                prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Re-raise the exception if it's unrelated to system role
                raise e

        return prompt


if __name__ == '__main__':
    cfg = COCOMConfig(decoder_model_name='mistralai/Mistral-7B-Instruct-v0.2',
                compr_model_name = "mistral_trimmed",
                compr_rate = 64,
                compr_n_layers = 5,
                compr_mlp_hidden_dim = 8096,
                compr_use_mlp = False, 
                lora = True, # lora on decoder (and decoder as compr)
                lora_compressor = True, # lora only on the compressor if it exists
                training_form = "both",
                load_adapters = True,
                kbtc_training = False,
                optimize_mem_tokens = True,
                different_mem_tokens = True,
                attn_implementation = 'flash_attention_2')
    
    cocom = COCOM(cfg)
    
    cocom.save_pretrained('test_ckpt')
    
    del cocom
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    cocom = COCOM.from_pretrained('test_ckpt')
