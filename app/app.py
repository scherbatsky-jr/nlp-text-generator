from flask import Flask, render_template, request, jsonify
import torch
from lib.lstm import LSTMLanguageModel

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = ""

    if request.method == 'POST':
        try:
            data = request.get_json()

            print(data)

            if data and 'prompt' in data:
                user_input = data['prompt']
                generated_text = process_user_input(user_input)

                return jsonify({'generated_text': generated_text})

            else:
                return jsonify({'error': 'Invalid JSON data'})

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index.html')

def process_user_input(user_input):
    max_seq_len = 30
    seed = 0
    temperature = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = torch.load('models/vocab.pt')
    tokenizer = torch.load('models/tokenizer.pt')
    vocab_size = len(vocab)
    emb_dim = 1024               
    hid_dim = 1024                
    num_layers = 2 
    dropout_rate = 0.65              
    lr = 1e-3

    model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
    model.load_state_dict(torch.load('models/harrypotter.pt',  map_location=device))

    generation = generate(user_input, max_seq_len, temperature, model, tokenizer, 
                          vocab, device, seed)
    return ' '.join(generation)

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction) 

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


if __name__ == '__main__':
    app.run(debug=True)
