IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_LENGTH = 576
MINIGPT4_IMAGE_TOKEN_LENGTH = 32
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
SHIKRA_IMAGE_TOKEN_LENGTH = 256
SHIKRA_IMG_START_TOKEN = 32001
SHIKRA_IMG_END_TOKEN = 32002

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "internvl": "USER: <ImageHere> <question> ASSISTANT:",
}

INSTRUCTION_TEMPLATE_NO_IMG = {
    "minigpt4": "###Human:<question> ###Assistant:",
    "instructblip": "<question>",
    "lrv_instruct": "###Human: <question> ###Assistant:",
    "shikra": "USER: <question> ASSISTANT:",
    "llava-1.5": "USER: <question> ASSISTANT:",
    "internvl": "USER: <question> ASSISTANT:",
}

SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

POPE_CHAT_PATH = {
    "random": "./pope_coco/chat/coco_pope_chat_random.json",
    "popular": "./pope_coco/chat/coco_pope_chat_popular.json",
    "adversarial": "./pope_coco/chat/coco_pope_chat_adversarial.json",
}
