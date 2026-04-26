from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
from pydub import AudioSegment
import soundfile as sf
import io

app = Flask(__name__)
app.secret_key = "mental_health_2025"

# ====================== 模拟数据 ======================
USERS = {"user": "123456"}
USER_INFO = {"user": {"name": "测试用户"}}
AREA_DATA = [
    {"name": "北京", "rate": "4.8%", "doctor": "张教授｜北京大学六院"},
    {"name": "上海", "rate": "5.2%", "doctor": "李主任｜上海精神卫生中心"},
    {"name": "广州", "rate": "4.5%", "doctor": "王医师｜中山三院"},
    {"name": "深圳", "rate": "5.0%", "doctor": "刘博士｜深圳康宁医院"},
    {"name": "杭州", "rate": "4.2%", "doctor": "陈医师｜浙大一院"},
    {"name": "成都", "rate": "4.6%", "doctor": "赵教授｜华西医院"},
]
ARTICLES = [
    {"title": "拒绝精神内耗，5个方法拥抱积极生活", "content": "每天给自己10分钟冥想时间，接纳情绪，拒绝自我否定..."},
    {"title": "科学缓解焦虑：从呼吸开始的身心调节", "content": "腹式呼吸法：吸气4秒，屏息4秒，呼气6秒，快速平复焦虑..."},
    {"title": "打破孤独感，建立健康的社交连接", "content": "从小型社交开始，每周一次深度交流，远离精神内耗..."},
]
FORUM_DATA = []
TEST_RECORDS = {}

# ====================== 算法核心（全格式音频兼容）======================
def load_audio(file, sr=16000):
    """
    双解析方案：兼容 WAV/MP3/M4A/OGG/FLAC 所有音频格式
    """
    file_bytes = file.read()
    file.seek(0)
    
    try:
        audio, _ = librosa.load(io.BytesIO(file_bytes), sr=sr, duration=5)
    except Exception:
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(file_bytes))
            wav_bytes = io.BytesIO()
            audio_segment.export(wav_bytes, format="wav")
            wav_bytes.seek(0)
            audio, _ = sf.read(wav_bytes)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = librosa.resample(audio, orig_sr=audio_segment.frame_rate, target_sr=sr)
        except Exception as e:
            raise Exception(f"音频解析失败，请检查文件格式: {str(e)}")
    
    max_len = 5 * sr
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
    return audio, sr

def extract_acoustic_features(audio, sr):
    zcr = librosa.feature.zero_crossing_rate(audio).mean()
    f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_mean = np.mean(f0[f0 > 0]) if len(f0[f0 > 0]) > 0 else 0
    rms = librosa.feature.rms(y=audio).mean()
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_mean = np.mean(mel_spec)
    return {
        "zero_cross_rate": float(round(zcr, 4)),
        "f0_mean": float(round(f0_mean, 2)),
        "rms_energy": float(round(rms, 4)),
        "mel_spec_mean": float(round(mel_mean, 4))
    }

def assess_depression_risk(feats):
    score = 0
    if feats["zero_cross_rate"] < 0.05: score +=20
    if feats["f0_mean"] < 120: score +=25
    if feats["rms_energy"] < 0.02: score +=25
    if feats["mel_spec_mean"] < 0.01: score +=30
    risk_score = min(score, 100)
    
    if risk_score < 30:
        level = "低风险"
        color = "text-emerald-400"
        sug = "心理状态优良，情绪稳定，保持健康作息。"
    elif risk_score < 60:
        level = "中风险"
        color = "text-amber-400"
        sug = "存在轻微情绪低落，建议多运动、放松心情。"
    else:
        level = "高风险"
        color = "text-rose-400"
        sug = "心理压力较高，建议及时寻求专业心理疏导。"
    return int(risk_score), level, color, sug

def plot_wave(audio):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 4), dpi=100)
    plt.plot(audio, color='#38bdf8', linewidth=1.2, alpha=0.9)
    plt.grid(alpha=0.1)
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img

# ====================== 全局布局 ======================
BASE_LAYOUT = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}} | 声愈·心镜</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {theme:{extend:{colors:{primary:'#38bdf8',darkbg:'#0b1120'}}}}
    </script>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;}
        body{background:#0b1120;color:#e2e8f0;overflow-x:hidden;}
        .glass{background:rgba(30,41,59,0.65);backdrop-filter:blur(12px);border:1px solid rgba(148,163,184,0.15);position:relative;overflow:hidden;}
        .glass::before{content:"";position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:conic-gradient(#38bdf8,#22d3ee,#a855f7,#38bdf8);animation:stream 4s linear infinite;z-index:-1;opacity:0.15;}
        @keyframes stream{100%{transform:rotate(360deg);}}
        .card-hover{transition:all 0.4s;}
        .card-hover:hover{transform:translateY(-8px);box-shadow:0 15px 35px rgba(56,189,248,0.2);}
        .fade-in-up{opacity:0;transform:translateY(40px);transition:all 0.8s;}
        .fade-in-up.show{opacity:1;transform:translateY(0);}
        .nav-item{transition:all 0.3s;}
        .nav-item:hover{color:#38bdf8;transform:translateY(-2px);}
        video{object-fit:cover;}
    </style>
</head>
<body>
    <nav class="glass fixed top-0 w-full z-50 py-4">
        <div class="container mx-auto px-6 flex justify-between items-center">
            <div class="flex items-center gap-3">
                <i class="fa-solid fa-brain text-primary text-2xl"></i>
                <h1 class="text-xl font-bold text-white">声愈·心镜</h1>
            </div>
            <div class="hidden md:flex gap-6 text-slate-300">
                <a href="/" class="nav-item">首页</a>
                <a href="/science" class="nav-item">心理科普</a>
                <a href="/test" class="nav-item">语音测试</a>
                <a href="/forum" class="nav-item">交流论坛</a>
                {% if session.user %}
                <a href="/profile" class="nav-item">个人中心</a>
                <a href="/logout" class="nav-item text-rose-400">退出</a>
                {% else %}
                <a href="/login" class="nav-item">登录</a>
                {% endif %}
            </div>
        </div>
    </nav>
    <main class="pt-24 pb-16 container mx-auto px-6 max-w-7xl">
        {{content|safe}}
    </main>
    <footer class="glass py-4 text-center text-slate-500 text-sm">© 2025 声愈·心镜 心理健康平台</footer>
    <script>
        document.addEventListener('DOMContentLoaded',()=>{
            const els = document.querySelectorAll('.fade-in-up');
            const ob = new IntersectionObserver(e=>{e.forEach(i=>i.isIntersecting&&i.target.classList.add('show'))});
            els.forEach(e=>ob.observe(e));
        });
    </script>
</body>
</html>
'''

# ====================== 1. 首页 ======================
@app.route('/')
def index():
    content = '''
    <div class="w-full h-[500px] rounded-2xl overflow-hidden mb-10 fade-in-up relative">
        <video autoplay muted loop class="w-full h-full absolute">
            <source src="https://assets.mixkit.co/videos/preview/mixkit-tree-with-yellow-flowers-1173-large.mp4" type="video/mp4">
        </video>
        <div class="absolute w-full h-full bg-black/50 flex flex-col items-center justify-center">
            <h1 class="text-5xl font-bold text-white mb-4">用声音守护心灵</h1>
            <p class="text-xl text-slate-300">AI语音心理筛查 · 让心理健康触手可及</p>
        </div>
    </div>

    <div class="glass p-10 rounded-2xl mb-10 fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-6"><i class="fa-solid fa-map-location-dot text-primary mr-2"></i>全国抑郁症分布率</h2>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-6">
            ''' + ''.join([f'''
            <div class="glass p-6 card-hover rounded-xl">
                <h3 class="font-bold text-xl text-white">{item["name"]}</h3>
                <p class="text-primary text-2xl my-2">{item["rate"]}</p>
                <p class="text-slate-400 text-sm">{item["doctor"]}</p>
            </div>
            ''' for item in AREA_DATA]) + '''
        </div>
    </div>

    <div class="glass p-10 rounded-2xl card-hover fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-6">平台核心功能</h2>
        <div class="grid md:grid-cols-4 gap-6">
            <div class="glass p-6 text-center card-hover rounded-xl"><i class="fa-solid fa-microphone text-primary text-3xl mb-3"></i><h3>语音测试</h3></div>
            <div class="glass p-6 text-center card-hover rounded-xl"><i class="fa-solid fa-book text-primary text-3xl mb-3"></i><h3>心理科普</h3></div>
            <div class="glass p-6 text-center card-hover rounded-xl"><i class="fa-solid fa-users text-primary text-3xl mb-3"></i><h3>交流论坛</h3></div>
            <div class="glass p-6 text-center card-hover rounded-xl"><i class="fa-solid fa-chart-line text-primary text-3xl mb-3"></i><h3>趋势分析</h3></div>
        </div>
    </div>
    '''
    return render_template_string(BASE_LAYOUT, title="首页", content=content, session=session)

# ====================== 2. 心理科普 ======================
@app.route('/science')
def science():
    content = '''
    <div class="glass p-10 rounded-2xl mb-10 fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-6"><i class="fa-solid fa-book-open text-primary mr-2"></i>心理科普文章</h2>
        <div class="space-y-6">
            ''' + ''.join([f'''
            <div class="glass p-6 card-hover rounded-xl">
                <h3 class="text-xl font-bold text-white">{item["title"]}</h3>
                <p class="text-slate-300 mt-2">{item["content"]}</p>
            </div>
            ''' for item in ARTICLES]) + '''
        </div>
    </div>
    '''
    return render_template_string(BASE_LAYOUT, title="心理科普", content=content, session=session)

# ====================== 3. 语音测试页面 ======================
@app.route('/test')
def test():
    if not session.get("user"):
        return redirect("/login")
    content = '''
    <div class="grid md:grid-cols-2 gap-8 mb-10 fade-in-up">
        <div class="glass rounded-2xl p-8 card-hover">
            <h3 class="text-xl font-bold text-white mb-4"><i class="fa-solid fa-microphone text-primary mr-2"></i>语音上传测试</h3>
            <div class="flex flex-col gap-4">
                <input type="file" id="audioFile" accept="audio/*" class="file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-primary/10 file:text-primary text-slate-200">
                <button onclick="analyzeAudio()" class="bg-primary text-slate-900 py-3 rounded-lg font-bold">开始AI分析</button>
            </div>
            <div id="loader" class="hidden mt-6 flex items-center gap-2 text-primary"><i class="fa-solid fa-spinner fa-spin"></i><span>分析中...</span></div>
        </div>
        <div class="glass rounded-2xl p-8 card-hover">
            <h3 class="text-xl font-bold text-white mb-4"><i class="fa-solid fa-chart-line text-primary mr-2"></i>测试结果</h3>
            <div id="result" class="hidden space-y-4">
                <div class="p-4 bg-slate-800/50 rounded-lg">
                    <p>等级：<span id="level" class="font-bold"></span></p>
                    <p>分数：<span id="score" class="text-primary font-bold"></span>/100</p>
                </div>
                <p id="suggest" class="text-slate-300"></p>
            </div>
        </div>
    </div>
    <div class="glass rounded-2xl p-8 fade-in-up">
        <h3 class="text-xl font-bold text-white mb-4">音频波形</h3>
        <img id="wave" class="w-full rounded-xl hidden">
    </div>
    <script>
        async function analyzeAudio(){
            const file = document.getElementById('audioFile').files[0];
            if(!file){alert('请选择音频');return;}
            document.getElementById('loader').classList.remove('hidden');
            try {
                let fd = new FormData(); fd.append('file',file);
                let res = await fetch('/upload',{method:'POST',body:fd});
                let data = await res.json();
                
                if (data.error) {
                    alert('分析失败：' + data.error);
                    return;
                }

                document.getElementById('level').innerText = data.level;
                document.getElementById('level').className = data.color;
                document.getElementById('score').innerText = data.score;
                document.getElementById('suggest').innerText = data.suggest;
                document.getElementById('wave').src = 'data:image/png;base64,' + data.wave;
                document.getElementById('result').classList.remove('hidden');
                document.getElementById('wave').classList.remove('hidden');
                
                fetch('/save_record?score=' + data.score + '&level=' + data.level);
            } catch (e) {
                alert('请求出错：' + e.message);
            } finally {
                document.getElementById('loader').classList.add('hidden');
            }
        }
    </script>
    '''
    return render_template_string(BASE_LAYOUT, title="语音测试", content=content, session=session)

# ====================== 4. 交流论坛 ======================
@app.route('/forum')
def forum():
    content = '''
    <div class="glass p-10 rounded-2xl mb-10 fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-6"><i class="fa-solid fa-comments text-primary mr-2"></i>交流论坛</h2>
        <div class="text-slate-300 leading-loose space-y-6 text-lg mb-8">
            <p>心理疾病 ≠ 软弱，它和感冒发烧一样，是需要治疗的健康问题。</p>
            <p>求助不是懦弱，而是勇敢和智慧的表现。</p>
            <p>每一种情绪都值得被接纳，每一个心灵都值得被温柔对待。</p>
            <p class="text-primary font-bold">消除病耻感，从理解、接纳、拥抱自己开始！</p>
        </div>

        <h3 class="text-2xl font-bold text-white mb-4">匿名交流区</h3>
        <form action="/forum/add" method="post" class="mb-6">
            <textarea name="content" class="w-full p-4 rounded-lg bg-slate-800 border border-slate-700 text-white h-24" placeholder="匿名发表你的心情..." required></textarea>
            <button class="mt-3 bg-primary text-slate-900 px-6 py-2 rounded-lg font-bold">发布</button>
        </form>
        <div class="space-y-4">
            ''' + ''.join([f'''
            <div class="glass p-4 rounded-xl">
                <p class="text-slate-300">{item["content"]}</p>
                <p class="text-xs text-slate-500 mt-2">{item["time"]} | 匿名用户</p>
            </div>
            ''' for item in FORUM_DATA[::-1]]) + '''
        </div>
    </div>
    '''
    return render_template_string(BASE_LAYOUT, title="交流论坛", content=content, session=session)

@app.route('/forum/add', methods=['POST'])
def forum_add():
    content = request.form.get('content')
    FORUM_DATA.append({"content": content, "time": datetime.now().strftime("%m-%d %H:%M")})
    return redirect('/forum')

# ====================== 5. 登录 ======================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS and USERS[username] == password:
            session['user'] = username
            return redirect('/profile')
        return "<script>alert('账号或密码错误！');location.href='/login'</script>"
    content = '''
    <div class="max-w-md mx-auto glass p-10 rounded-2xl fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-6 text-center">用户登录</h2>
        <form method="post" class="space-y-4">
            <input name="username" class="w-full p-3 rounded-lg bg-slate-800 border border-slate-700 text-white" placeholder="用户名" required>
            <input name="password" type="password" class="w-full p-3 rounded-lg bg-slate-800 border border-slate-700 text-white" placeholder="密码" required>
            <button class="w-full bg-primary text-slate-900 py-3 rounded-lg font-bold">登录</button>
            <p class="text-center text-slate-400">没有账号？<a href="/register" class="text-primary">立即注册</a></p>
        </form>
    </div>
    '''
    return render_template_string(BASE_LAYOUT, title="登录", content=content, session=session)

# ====================== 6. 注册 ======================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_pwd = request.form.get('confirm_pwd')
        name = request.form.get('name')
        if not username or not password or not confirm_pwd or not name:
            return "<script>alert('请填写完整信息！');location.href='/register'</script>"
        if password != confirm_pwd:
            return "<script>alert('两次输入的密码不一致！');location.href='/register'</script>"
        if username in USERS:
            return "<script>alert('用户名已存在，请更换！');location.href='/register'</script>"
        USERS[username] = password
        USER_INFO[username] = {"name": name}
        TEST_RECORDS[username] = []
        return "<script>alert('注册成功！请登录');location.href='/login'</script>"
    content = '''
    <div class="max-w-md mx-auto glass p-10 rounded-2xl fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-6 text-center">用户注册</h2>
        <form method="post" class="space-y-4">
            <input name="username" class="w-full p-3 rounded-lg bg-slate-800 border border-slate-700 text-white" placeholder="用户名" required>
            <input name="password" type="password" class="w-full p-3 rounded-lg bg-slate-800 border border-slate-700 text-white" placeholder="密码" required>
            <input name="confirm_pwd" type="password" class="w-full p-3 rounded-lg bg-slate-800 border border-slate-700 text-white" placeholder="确认密码" required>
            <input name="name" class="w-full p-3 rounded-lg bg-slate-800 border border-slate-700 text-white" placeholder="姓名" required>
            <button class="w-full bg-primary text-slate-900 py-3 rounded-lg font-bold">注册</button>
            <p class="text-center text-slate-400">已有账号？<a href="/login" class="text-primary">立即登录</a></p>
        </form>
    </div>
    '''
    return render_template_string(BASE_LAYOUT, title="注册", content=content, session=session)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ====================== 7. 个人中心 ======================
@app.route('/profile')
def profile():
    if not session.get("user"):
        return redirect("/login")
    username = session['user']
    if username not in TEST_RECORDS:
        TEST_RECORDS[username] = []
    high_risk = [r for r in TEST_RECORDS[username] if r['level']=="高风险"]
    user_info = USER_INFO[username]
    content = f'''
    <div class="glass p-10 rounded-2xl mb-6 fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-4">个人中心</h2>
        <p class="text-xl">用户名：{username}</p>
        <p class="text-xl">姓名：{user_info["name"]}</p>
    </div>

    <div class="glass p-6 rounded-2xl mb-6 fade-in-up {'bg-rose-800/20' if high_risk else ''}">
        <h3 class="text-xl font-bold text-white mb-2"><i class="fa-solid fa-bell text-rose-400 mr-2"></i>风险预警</h3>
        <p class="{'text-rose-400 font-bold' if high_risk else 'text-emerald-400'}">
            {'⚠️ 检测到高风险心理状态，请立即寻求专业帮助！' if high_risk else '✅ 当前心理状态稳定'}
        </p>
    </div>

    <div class="grid md:grid-cols-3 gap-6 mb-10">
        <a href="/trend" class="glass p-6 card-hover rounded-xl text-center"><i class="fa-solid fa-line-chart text-primary text-3xl mb-3"></i><h3>趋势分析</h3></a>
        <a href="/settings" class="glass p-6 card-hover rounded-xl text-center"><i class="fa-solid fa-gear text-primary text-3xl mb-3"></i><h3>账号设置</h3></a>
        <a href="/test" class="glass p-6 card-hover rounded-xl text-center"><i class="fa-solid fa-microphone text-primary text-3xl mb-3"></i><h3>开始测试</h3></a>
    </div>
    '''
    return render_template_string(BASE_LAYOUT, title="个人主页", content=content, session=session)

# ====================== 8. 设置页面 ======================
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if not session.get("user"):
        return redirect("/login")
    username = session['user']
    if request.method == 'POST':
        USER_INFO[username]["name"] = request.form.get('name')
        return "<script>alert('修改成功！');location.href='/settings'</script>"
    user_info = USER_INFO[username]
    content = f'''
    <div class="glass p-10 rounded-2xl fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-6">账号设置</h2>
        <form method="post" class="space-y-4 max-w-md">
            <input name="name" value="{user_info['name']}" class="w-full p-3 rounded-lg bg-slate-800 text-white">
            <button class="bg-primary text-slate-900 px-6 py-2 rounded-lg">保存修改</button>
        </form>
        <div class="mt-6 p-4 glass rounded-xl">
            <h3 class="font-bold text-white mb-2">隐私保护</h3>
            <p class="text-slate-300">所有语音数据仅本地分析，绝不上传泄露</p>
        </div>
    </div>
    '''
    return render_template_string(BASE_LAYOUT, title="账号设置", content=content, session=session)

# ====================== 9. 趋势分析（显示风险等级）======================
@app.route('/trend')
def trend():
    if not session.get("user"):
        return redirect("/login")
    username = session['user']
    if username not in TEST_RECORDS:
        TEST_RECORDS[username] = []
    records = TEST_RECORDS[username]
    content = f'''
    <div class="glass p-10 rounded-2xl fade-in-up">
        <h2 class="text-3xl font-bold text-white mb-6">心理状态趋势分析</h2>
        <div class="glass p-6 rounded-xl mb-6">
            <p class="text-slate-300">系统持续追踪你的语音测试分数，生成心理状态变化曲线，提前预警异常波动</p>
        </div>

        <div class="space-y-4">
            <h3 class="text-2xl font-bold text-white mb-4">历史测试记录</h3>
            {''.join([f'''
            <div class="glass p-4 rounded-xl flex justify-between items-center">
                <div>
                    <p class="text-white font-bold">{r['time']}</p>
                    <p class="text-slate-400">分数：{r['score']}/100</p>
                </div>
                <p class="font-bold text-lg {
                    'text-emerald-400' if r['level'] == '低风险' else 
                    'text-amber-400' if r['level'] == '中风险' else 
                    'text-rose-400'
                }">{r['level']}</p>
            </div>
            ''' for r in records[::-1]]) if records else '<p class="text-slate-400">暂无测试记录，快去做一次语音测试吧！</p>'}
        </div>
    </div>
    '''
    return render_template_string(BASE_LAYOUT, title="趋势分析", content=content, session=session)

# ====================== 保存测试记录 ======================
@app.route('/save_record')
def save_record():
    if not session.get("user"):
        return "未登录"
    username = session['user']
    score = int(request.args.get('score'))
    level = request.args.get('level')
    if username not in TEST_RECORDS:
        TEST_RECORDS[username] = []
    TEST_RECORDS[username].append({
        "score": score, 
        "level": level, 
        "time": datetime.now().strftime("%m-%d %H:%M")
    })
    return "ok"

# ====================== 音频上传接口 ======================
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        audio, sr = load_audio(file)
        feats = extract_acoustic_features(audio, sr)
        score, level, color, sug = assess_depression_risk(feats)
        wave_img = plot_wave(audio)
        return jsonify({"features": feats,"score": score,"level": level,"color": color,"suggest": sug,"wave": wave_img})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)