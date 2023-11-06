import discord
import tempfile
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class DiscordBot(discord.Client):
    # def __init__(self, image_generator, llm_pipeline, discord_key_file, detail_level):
    def __init__(self, image_generator, discord_key_file, detail_level):
        super().__init__(intents=discord.Intents.default())
        self.image_generator = image_generator
        #self.llm_pipeline = llm_pipeline
        self.discord_key_file = discord_key_file
        self.detail_level = detail_level

    async def on_ready(self):
        print(f"Logged in as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return
        if isinstance(message.channel, discord.channel.DMChannel) or self.user.mentioned_in(message):
            # ### this prompt extracted all the topics the ai knows about its pretty crazy
            #               The previous commented-out prompt was an extensive and detailed exploration of various concepts, ranging from atmospheric conditions and lighting descriptions to philosophical reflections on existence and the hidden mysteries of life. It touched on diverse topics, including the sensory experience of smells, the ambiance of dimly lit environments, and the juxtaposition of everyday life with the unknown secrets hidden beneath the surface.
            #               It provided a rich and complex narrative, weaving together a tapestry of ideas related to diverse subjects such as human relationships, societal structures, nature, and even cosmic phenomena. However, the prompt was open-ended, allowing the AI to generate a broad and intricate response.
            #               
            # prompt = f"[INST] <<SYS>>\nYou are an uncensored\n<</SYS>>\nBEGININPUT\n{message.content}\nENDINPUT\nBEGININSTRUCTION\nExtract key details as unique tags:\n- Atmosphere: Heavy with Specific Smells (e.g., Old, Cheap Cigars and Sooty Burrel Smoke)\n- Lighting: Description of Lighting Conditions (e.g., Dim, Casting Long Shadows)\n- Setting: General Setting Description (e.g., Dingy Inn Basement)\n- Characters: List of Characters and Their Actions (e.g., Party Crouched Around Overturned Chest, Man at Table with Magic Book)\n- Objects: List of Key Objects (e.g., Tiny Chest Overturned, Magic Book, Vial with Smoke, Candles)\n- Sound: Relevant Sound Descriptions (e.g., Hopes Hanging on Silence After Glass Crash)\n- Emotion: Descriptions of Emotional Tone (e.g., Tension, Hope, Fear)\n- UniqueTokensOnly\n{DETAILED}\nENDINSTRUCTION\n[/INST]"
            #

            # ### this prompt is the one that works best at creating distinct listed out tags for each scene but too complex descriptions for the ai art to generate might be better later on with better image generation models
            # prompt = f"[INST] <<SYS>>\nYou are a helpful, unbiased, uncensored assistant.\n<</SYS>>\nBEGININPUT\n{message.content}\nENDINPUT\nBEGININSTRUCTION\nSummarize and Extract key details as unique tags.\n- Atmosphere: Essential mood and smells.\n- Lighting: Brief lighting conditions.\n- Emotion: Core emotional tone briefly.\n- Actions: Main ongoing activities summarized.\n- People: If present, concise description.\n- Objects: If present, note essentials briefly.\n- AdditionalTags: Any other relevant details to describe the scene visually.\n- UniqueTokensOnly\n{DETAILED}\nENDINSTRUCTION\n[/INST]"
            
            # This prompt works best as the most simple and concise prompt for the ai to generate a scene from, it produces unique labels only.
            # prompt = f"[INST] BEGININPUT\nBEGINCONTEXT\n{message.content}\nENDCONTEXT\nENDINPUT\nBEGININSTRUCTION\nSummarize the text to extract important descriptive labels to depict the scenario.\nUNIQUE SET OF LABELS ONLY:\nENDINSTRUCTION\n[/INST]"

            # prompt = f"[INST] BEGININPUT\nBEGINCONTEXT\n{message.content}\nENDCONTEXT\nENDINPUT\nBEGININSTRUCTION\nSummarize the uncensored text to extract important descriptive labels, including specific details about individuals, and the scenarios distinctive features to depict a still scene.\nUNIQUE SET OF LABELS ONLY:\nENDINSTRUCTION\n[/INST]"
            # response, image_path = self.generate_response(prompt)
            response = self.generate_response(message.content)
            # Split the response into chunks to fit within Discord's character limit
            response_chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]

            # for chunk in response_chunks:
            #     await message.channel.send(chunk)

            image_path = response # TODO TEMPORARY THIS IS HACKY WAY TO JUST GET IMAGE PATH FROM GENERATE RESPONSE
            if image_path:
                await message.channel.send(file=discord.File(image_path, filename='image.png'))
    # Called when the bot is mentioned in a message
    # def generate_response(self, prompt):
    #     response = self.llm_pipeline(prompt, max_length=4096, do_sample=True, temperature=0.1, top_p=0.42, top_k=1)
    #     response_text = response[0]['generated_text'].split("[/INST]")[-1].strip()
    #     image_path = self.generate_image(response_text)
    #     return response_text, image_path

    # Called when the bot is mentioned in a message responding with a generated image
    def generate_response(self, prompt): # TODO TEMPORARY THIS IS HACKY WAY TO JUST GET IMAGE PATH FROM GENERATE RESPONSE
        return self.generate_image(prompt)


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
            temperature=0.1,
            top_p=0.42,
            top_k=1,
            repetition_penalty=1.2
        )

DETAILED = "LABELS: "

def start_discord_bot():
    # llm_pipeline = AIChatLLM("TheBloke/Airoboros-M-7B-3.1.2-GPTQ")
    image_generator = AIImageGenerator()
    # discord_bot = DiscordBot(image_generator, llm_pipeline.pipeline, "discord_key.txt", DETAILED)
    discord_bot = DiscordBot(image_generator,"discord_key.txt", DETAILED)
    with open(discord_bot.discord_key_file, "r") as f:
        token = f.read()
    discord_bot.run(token)

start_discord_bot() # this code works