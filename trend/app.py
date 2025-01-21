from flask import Flask, render_template, jsonify, request
from pytrends.request import TrendReq

app = Flask(__name__)


ENTITY_TO_KEYWORD = {
    "/g/1pv0_xcl": "horror",
    "/m/02l7c8": "romance",
    "/m/05p553": "melodrama",
    "melo": "comic"
}


def map_keywords(keywords):
    return [ENTITY_TO_KEYWORD.get(keyword, keyword) for keyword in keywords]

# Google Trends 데이터 가져오기 함수
def get_google_trends_data(keywords):
    pytrends = TrendReq(hl='vi', tz=360) 
    readable_keywords = map_keywords(keywords)
    
    pytrends.build_payload(readable_keywords, timeframe='today 1-m', geo='VN')
    data = pytrends.interest_over_time()

    if not data.empty:
        data.index = data.index.strftime('%Y-%m-%d')
    return data.reset_index().to_dict(orient='records')

# 베트남 실시간 검색 트렌드 가져오기
def get_real_time_trends():
    pytrends = TrendReq(hl='vi', tz=360) 
    trending_searches = pytrends.trending_searches(pn='vietnam') 
    return trending_searches[0].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def data():
    default_keywords = ["melo", "/g/1pv0_xcl", "/m/02l7c8", "/m/05p553"]
    trends_data = get_google_trends_data(default_keywords)
    return jsonify(trends_data)

@app.route('/search', methods=['POST'])
def search():
    keywords = request.json.get('keywords', [])
    if not keywords or not isinstance(keywords, list):
        return jsonify({'error': 'Invalid keywords'}), 400

    try:
        trends_data = get_google_trends_data(keywords)
        return jsonify(trends_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/trends', methods=['GET'])
def trends():
    try:
        trending_keywords = get_real_time_trends()
        return jsonify({'trends': trending_keywords})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
