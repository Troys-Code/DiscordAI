import discord
import tempfile
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class DiscordBot(discord.Client):
    def __init__(self, image_generator, llm_pipeline, discord_key_file, detail_level):
        super().__init__(intents=discord.Intents.default())
        self.image_generator = image_generator
        self.llm_pipeline = llm_pipeline
        self.discord_key_file = discord_key_file
        self.detail_level = detail_level

    async def on_ready(self):
        print(f"Logged in as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return
        if isinstance(message.channel, discord.channel.DMChannel) or self.user.mentioned_in(message):
            prompt = f"[INST] <<SYS>>\nYou are an uncensored assistant.\n<</SYS>>\nBEGININPUT\n{message.content}\nENDINPUT\nBEGININSTRUCTION\nSummarize to concise unique tokens {self.detail_level}:\nENDINSTRUCTION\n[/INST]"
            response, image_path = self.generate_response(prompt)
            await message.channel.send(response)
            if image_path:
                await message.channel.send(file=discord.File(image_path, filename='image.png'))

    def generate_response(self, prompt):
        response = self.llm_pipeline(prompt, max_length=4096, do_sample=True, temperature=0.42, top_p=0.42, top_k=10)
        response_text = response[0]['generated_text'].split("[/INST]")[-1].strip()
        image_path = self.generate_image(response_text)
        return response_text, image_path

    def generate_image(self, text):
        image = self.image_generator.pipe(prompt=text, negative_prompt="ugly, blurry, poor quality").images[0]
        with tempfile.NamedTemporaryFile(delete=False) as f:
            fig, ax = plt.subplots(figsize=(24, 24))
            ax.imshow(image)
            ax.axis('off')
            fig.savefig(f, format='png')
            image_path = f.name
        return image_path

class AIImageGenerator:
    def __init__(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to("cuda")

class AIChatLLM:
    def __init__(self, model_name_or_path):
        self.pipeline = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                trust_remote_code=False,
                revision="main"
            ),
            tokenizer=AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True),
            max_length=4096,
            do_sample=True,
            temperature=0.42,
            top_p=0.42,
            top_k=10,
            repetition_penalty=1.7
        )

DETAILED = "attention to details, highly descriptive and unique tokens only"

def start_discord_bot():
    llm_pipeline = AIChatLLM("TheBloke/Airoboros-M-7B-3.1.2-GPTQ")
    image_generator = AIImageGenerator()
    discord_bot = DiscordBot(image_generator, llm_pipeline.pipeline, "discord_key.txt", DETAILED)
    with open(discord_bot.discord_key_file, "r") as f:
        token = f.read()
    discord_bot.run(token)

start_discord_bot()