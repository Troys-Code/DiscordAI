from setuptools import setup, find_packages

setup(
    name='discord-ai-bot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib',
        'diffusers',
        'transformers',
        'discord.py',
    ],
    entry_points={
        'console_scripts': [
            'start-discord-bot=DiscordAI_Bot:start_discord_bot',
        ],
    },
)
