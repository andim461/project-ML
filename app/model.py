from transformers import PretrainedConfig
import pytorch_lightning as pl
from transformers import MBartConfig
from transformers import MBart50TokenizerFast
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.mbart import MBartForConditionalGeneration
from transformers.optimization import AdamW, get_constant_schedule_with_warmup
import torch


tokenizer = MBart50TokenizerFast.from_pretrained("./tokenizer",src_lang='ru_RU', tgt_lang='ru_RU')

class Seq2SeqConfig(PretrainedConfig):
    
    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]
   
    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        gradient_checkpointing=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        omit_encoder_input_embeddings=False,
        separate_input_vocab=False,
        separate_input_vocab_size=None,
        separate_input_padding_idx=0,
        position_embedding_type='learned',
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.omit_encoder_input_embeddings = omit_encoder_input_embeddings

        # Добавим опцию для отдельного словаря перед Encoder.
        assert bool(separate_input_vocab) == bool(separate_input_vocab_size)
        self.separate_input_vocab = separate_input_vocab
        self.separate_input_vocab_size = separate_input_vocab_size
        self.separate_input_padding_idx = separate_input_padding_idx
        self.position_embedding_type = position_embedding_type
        #
        # if self.separate_input_vocab:
        #     assert not self.tie_word_embeddings

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model

config = Seq2SeqConfig(
    omit_encoder_input_embeddings=True,
    separate_input_vocab_size=tokenizer.vocab_size,
    separate_input_padding_idx=tokenizer.pad_token_id,
    separate_input_vocab=True,
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    max_length=20,
    encoder_layers=3,
    decoder_layers=3,

    use_cache=False,
    decoder_ffn_dim = 1024,
    encoder_ffn_dim = 1024,
)

hparams = {
    'huggingface_config':config.__dict__,
    'learning_rate': 5e-4,
    'warmup_steps': 500,
}


class BartConditionalGeneration(pl.LightningModule):

    def __init__(self, hparams):
      super().__init__()
      config_class = MBartConfig
      config = config_class(**hparams['huggingface_config'])
      model_class = MBartForConditionalGeneration
      self.seq2seq = model_class(config)
      self.save_hyperparameters(hparams)


    def forward_with_labels(self, batch):
        
        assert isinstance(batch, dict)
        assert 'labels' in batch and batch['labels'] is not None
        
        
        outputs: Seq2SeqLMOutput = self.seq2seq(**batch, return_dict=True)
        
        return outputs

    def training_step(self, batch, batch_idx):
        
        outputs: Seq2SeqLMOutput = self.forward_with_labels(batch)
        
        self.log('step_loss', outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs: Seq2SeqLMOutput = self.forward_with_labels(batch)
        self.log('val_step_loss', outputs.loss, prog_bar=True)
        return {
            'val_step_loss': outputs.loss.item()
        }

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = sum(d['val_step_loss'] for d in outputs) / len(outputs)
        self.log('avg_val_loss', avg_loss, prog_bar=True)

    def configure_optimizers(self):
        
        optimizer = AdamW(params=self.parameters(), lr=self.hparams['learning_rate'])
        warmup_steps = self.hparams['warmup_steps']
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step'
        }]

model_new = BartConditionalGeneration(hparams) 
model_new.load_state_dict(torch.load('./model'))

def model_device(model):
    return next(iter(model.parameters())).device

def generate(model, input_ids, input_attention_mask, **gen_args):
    model.eval()
    device = model_device(model)
    input_ids, input_attention_mask = input_ids.to(device), input_attention_mask.to(device)
    
    
    # Да, EOS а не BOS, новая мода
    decoder_start_token = model.seq2seq.config.eos_token_id
    # приходится эмбеддить руками, чтобы в encoder шли inputs_embeds
    
    inputs_embeds = model.seq2seq.get_input_embeddings()
    with torch.no_grad():
        encoder_outputs = model.seq2seq.get_encoder()(attention_mask=input_attention_mask, input_ids=input_ids)
        
        output = model.seq2seq.generate(encoder_outputs=encoder_outputs,
                               decoder_start_token_id=decoder_start_token,
                               **gen_args)
        
        return tokenizer.decode(output[0].tolist(), skip_special_tokens=True)


def generate_from_text(model, text, **gen_args):
    tokenized = tokenizer(text, return_tensors='pt')
    return generate(model, tokenized['input_ids'], tokenized['attention_mask'], **gen_args)