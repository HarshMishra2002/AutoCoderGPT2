from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import iframe

#from streamlit_option_menu import option_menu

if __name__ == '__main__':
    
    import streamlit as st

# Using object notation
    st.sidebar.title("Mini Project: ")
    user_menu = st.sidebar.radio(
    'Select an option',
    ('Generate Code', 'Model & Training Details', 'Installation Guide'))
    
    if user_menu == 'Generate Code':
        st.title("Python Code Generation using GPT2 Model")
        parser = argparse.ArgumentParser(description='Params')
        parser.add_argument('--model_path', type=str, default="model/gpt2_medium_fine_tuned_coder",
                            help='the path to load fine-tuned model')
        parser.add_argument('--max_length', type=int, default=128,
                            help='maximum length for code generation')
        parser.add_argument('--temperature', type=float, default=0.7,
                            help='temperature for sampling-based code geneeration')
        parser.add_argument(
            "--use_cuda", action="store_true", help="inference with gpu?"
        )

        args = parser.parse_args()

        # load fine-tunned model and tokenizer from path
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

        model.eval()
        if args.use_cuda:
            model.to("cuda")

        # now the fine-tunned model supports two programming languages, namely, python and java
        def lang_select():
            lang = ""
            while lang not in ["python", "java"]:
                print('Enter the programming language you prefer (python or java)')
                lang = input(">>> ").lower()
            return lang


        lang = "python"
        #lang = lang_select()

        #context = ""
        context = st.text_area("Enter the context code (type exit to stop)")
        max_length_input = st.number_input("Enter the Ouput length (type 128 for default)", value = 128)
        #while context != "exit":
            #print(f'You are using {lang} now. Enter the context code (exit or change_lang)')
            #context = st.text_input("Enter the context code (type exit to stop)")
            #context = input(">>> ")

        if context == "change_lang":
            lang = lang_select()

    #         print(f"You are using {lang} now. Enter the context code")
    #         context = input(">>> ")

        input_ids = tokenizer.encode("<python> " + context,
                                         return_tensors='pt') if lang == "python" else tokenizer.encode(
                "<java> " + context, return_tensors='pt')
        outputs = model.generate(input_ids=input_ids.to("cuda") if args.use_cuda else input_ids,
                                     max_length=max_length_input,
                                     temperature=args.temperature,
                                     num_return_sequences=1)
        for i in range(1):
            if st.button("Generate"):
                decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
                    # ends with occurence of double new lines (to meet the convention of code completion)
#                 if "\n\n" in decoded:

#                     decoded = decoded[:decoded.index("\n\n")]

                    #print('Generated {}: {}'.format(i, decoded))
                st.code(decoded)
        components.iframe(''' https://www.jdoodle.com/a/63kn ''',width=800, height=666 ,scrolling=True)
        
    if user_menu == 'Model & Training Details':
        
        st.title("Model & Training Details")
        st.subheader("The workflow of CoderGPT at training and inference time")
        st.image("workflow.png")
        st.subheader("Dataset Preperation")
        st.text("As a starting point, we will just focuses on The Algorithms library now.\nWe want coderGPT to help auto-complete codes at a general level.\nThe codes of The Algorithms suits the need! Also, think the codes \nfrom The Algorithms is well written (high-quality codes!). ")
        st.image("dataprocessing.png")
        st.code("parser.add_argument('--segment_len', type=int, default=254,help='the length of each example')\nparser.add_argument('--stride', type=int, default=10, help='stride to split training examples') \nparser.add_argument('--dev_size', type=float, default=0.1, help='split ratio of development set for each language')")
        st.code("gpt2_tok = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=False) \nfor path in paths: \n    source_files = glob.glob(f'{path}/**/*.py' if path == 'Python' else f'{path}/**/*.java', recursive=True) \n    for each_src in tqdm(source_files): \n        with open(each_src, 'r', encoding='utf-8') as f: \n            code_content = f.read() \n            encoded = gpt2_tok.encode(code_content) \n            for i in range(len(encoded) // args.stride): \n                seg = encoded[i * args.stride:i * args.stride + args.segment_len] \n                if path not in segments: \n                    segments[path] = [] \n                segments[path].append(json.dumps({'token_ids': seg, 'label': path}))")
        st.subheader("Pre-trained Model")
        st.code("class GPTSingleHead(nn.Module):\n    def __init__(self, model_name_or_path: str, max_seq_length: int = 256, do_lower_case: bool = False,special_words_to_add=None):\n        super(GPTSingleHead, self).__init__()\n        self.config_keys = ['max_seq_length', 'do_lower_case']\n        self.do_lower_case = do_lower_case\n        if max_seq_length > 1024:\n            logging.warning('GPT only allows a max_seq_length of 1024. Value will be set to 1024')\n            max_seq_length = 1024\n        self.max_seq_length = max_seq_length\n        self.gpt = GPT2LMHeadModel.from_pretrained(model_name_or_path)\n        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)\n        if special_words_to_add != None:\n            self.add_special_words(special_words_to_add)\n        self.bos_token_id=self.tokenizer.bos_token_id\n        self.eos_token_id=self.tokenizer.eos_token_id\n        # self.pad_token_id=self.tokenizer.pad_token_id")
        st.subheader("Finetuning")
        st.text("Before fine-tuning, we split the dataset into a train set and development set \nat a ratio of 9:1.This leads to ending up with around 69k training examples \nand 7k validation examples.")
        st.text("")
        st.text("The fine-tuning process trains the GPT2LMHeadModel in a batch size of 4 per GPU. We\nset the maximum sequence length to be 256 due to computational resources restrictions.")
        st.text("")
        st.text("Fine-tuning is done on one GPU2 (12GB RTX 2080Ti) with around 24 hours for \nfine-tuning gpt2-medium (approx. 86k total steps ). \nWe use Adam as the optimiser and set the learning rate to be 5e−5. \nDuring the fine-tuning, the best model saved is determined by perplexity evaluated \non the development set with evaluation step of 200.")
        st.image("trainresult.png")
if user_menu == 'Installation Guide':
    st.title("Installation Guide")
    st.title("")
    st.subheader("Download the Code from Github")
    st.code("git clone https://github.com/HarshMishra2002/AutoCoderGPT2.git")
    st.subheader("Install necessary Modules")
    st.code("pip install -r requirements.txt")
    st.subheader("Run the Interact file")
    st.code("python interact.py")