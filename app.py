# UNUSED IN FINAL PROJECT!

from flask import Flask, render_template, jsonify, request
# import bot
from chatbot.main import init_chatbot, evaluate_sentence

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        return render_template('chatbot.html')
    elif request.method == 'POST':
        try:
            datasets = request.json['datasets']
            voc, _, encoder, decoder, _, _, _, _ = init_chatbot(datasets=datasets, load_from_file=True)
            output_sentence = evaluate_sentence(encoder, decoder, voc, request.json['msg'])
            return jsonify(output_sentence)
            # input_sentence = bot.normalizeString(request.json['msg'])
            # # Evaluate sentence
            # output_words = bot.evaluate(bot.encoder, bot.decoder, bot.searcher, bot.voc, input_sentence)
            # # Format and print response sentence
            # output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            # return jsonify(request.json['output_words'])
        except KeyError:
            return jsonify("Error: Encountered unknown word.")
    # if request.method == 'POST':
    #     print(request)
    #     # msg = request.form.get('msg')
    #     # return msg


@app.route('/test')
def test():
    return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
