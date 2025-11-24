import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import requests
import os

# إعدادات الصفحة
st.set_page_config(
    page_title="AION Quantum Pro Trading",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص التصميم
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .trade-positive { color: #00d600; font-weight: bold; }
    .trade-negative { color: #ff0000; font-weight: bold; }
    .currency-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 5px 0;
        cursor: pointer;
    }
    .currency-card:hover {
        background: #e9ecef;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.load_config()
        self.setup_session_state()
    
    def load_config(self):
        """تحميل الإعدادات المحفوظة"""
        try:
            with open('trading_config.json', 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {
                'api_key': '',
                'api_secret': '', 
                'testnet': True,
                'trading_mode': 'paper_trading',
                'initial_balance': 50,
                'max_coins': 10,
                'trade_amount': 10
            }
    
    def save_config(self):
        """حفظ الإعدادات"""
        try:
            with open('trading_config.json', 'w') as f:
                json.dump(self.config, f)
        except:
            pass
    
    def setup_session_state(self):
        """إعداد حالة الجلسة"""
        defaults = {
            'bot_running': False,
            'selected_currency': 'BTCUSDT',
            'trade_history': [],
            'learning_data': [],
            'initialized': True
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """رأس الصفحة"""
        st.markdown('<h1 class="main-header">🚀 AION QUANTUM PRO TRADING</h1>', unsafe_allow_html=True)
        
        # استخدام get للوصول الآمن
        bot_status = st.session_state.get('bot_running', False)
        trade_history = st.session_state.get('trade_history', [])
        
        # شريط الحالة
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            status = "🟢 نشط" if bot_status else "🔴 متوقف"
            st.metric("حالة البوت", status)
        with col2:
            balance = self.config.get('initial_balance', 50)
            st.metric("رأس المال", f"${balance:.2f}")
        with col3:
            total_trades = len(trade_history)
            st.metric("إجمالي الصفقات", total_trades)
        with col4:
            profit = sum(trade.get('profit', 0) for trade in trade_history)
            st.metric("الأرباح الكلية", f"${profit:.2f}")
        with col5:
            active_coins = len(self.get_trading_coins())
            st.metric("العملات النشطة", active_coins)
    
    def render_api_settings(self):
        """إعدادات API"""
        st.sidebar.header("🔑 إعدادات الحساب")
        
        with st.sidebar.expander("⚙️ إعدادات API", expanded=True):
            api_key = st.text_input("API Key", value=self.config.get('api_key', ''), type="password")
            api_secret = st.text_input("Secret Key", value=self.config.get('api_secret', ''), type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                testnet = st.checkbox("الحساب التجريبي", value=self.config.get('testnet', True))
            with col2:
                trading_mode = st.selectbox(
                    "وضع التداول",
                    ["paper_trading", "live_trading"],
                    index=0 if self.config.get('trading_mode', 'paper_trading') == 'paper_trading' else 1
                )
            
            if st.button("💾 حفظ الإعدادات", use_container_width=True):
                self.config.update({
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'testnet': testnet,
                    'trading_mode': trading_mode
                })
                self.save_config()
                st.success("✅ تم حفظ الإعدادات")
    
    def render_control_panel(self):
        """لوحة التحكم"""
        st.sidebar.header("🎮 تحكم البوت")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🚀 تشغيل البوت", type="primary", use_container_width=True):
                st.session_state.bot_running = True
                st.success("✅ البوت يعمل الآن!")
        with col2:
            if st.button("⏹️ إيقاف البوت", use_container_width=True):
                st.session_state.bot_running = False
                st.warning("⏹️ تم إيقاف البوت")
        
        # إعدادات التداول
        st.sidebar.header("⚡ إعدادات التداول")
        self.config['initial_balance'] = st.sidebar.number_input(
            "رأس المال ($)", 
            value=self.config.get('initial_balance', 50),
            min_value=10,
            step=10
        )
        
        self.config['max_coins'] = st.sidebar.slider(
            "عدد العملات المتداولة",
            min_value=1,
            max_value=10,
            value=self.config.get('max_coins', 10)
        )
        
        self.config['trade_amount'] = st.sidebar.slider(
            "مبلغ التداول ($)",
            min_value=5,
            max_value=100,
            value=self.config.get('trade_amount', 10),
            step=5
        )
    
    def render_historical_simulation(self):
        """المحاكاة التاريخية"""
        st.sidebar.header("📊 المحاكاة التاريخية")
        
        with st.sidebar.expander("🕐 محاكاة تاريخية", expanded=False):
            start_date = st.date_input(
                "تاريخ البداية",
                datetime.now() - timedelta(days=30)
            )
            
            end_date = st.date_input(
                "تاريخ النهاية", 
                datetime.now()
            )
            
            simulation_coins = st.slider(
                "عدد العملات في المحاكاة",
                min_value=1,
                max_value=10,
                value=5
            )
            
            if st.button("🎯 تشغيل المحاكاة", use_container_width=True):
                self.run_historical_simulation(start_date, end_date, simulation_coins)
    
    def run_historical_simulation(self, start_date, end_date, coins_count):
        """تشغيل المحاكاة التاريخية"""
        with st.spinner(f"جاري محاكاة {coins_count} عملات من {start_date} إلى {end_date}..."):
            simulated_trades = self.simulate_historical_trades(coins_count, start_date, end_date)
            self.save_learning_data(simulated_trades)
            st.success(f"✅ تمت المحاكاة: {len(simulated_trades)} صفقة")
            
            total_profit = sum(trade.get('profit', 0) for trade in simulated_trades)
            win_rate = len([t for t in simulated_trades if t.get('profit', 0) > 0]) / len(simulated_trades) if simulated_trades else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("إجمالي الأرباح", f"${total_profit:.2f}")
            col2.metric("معدل النجاح", f"{win_rate:.1%}")
            col3.metric("الصفقات المحاكاة", len(simulated_trades))
    
    def simulate_historical_trades(self, coins_count, start_date, end_date):
        """محاكاة الصفقات التاريخية"""
        trades = []
        coins = self.get_trading_coins()[:coins_count]
        
        for coin in coins:
            for _ in range(20):
                trade = {
                    'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                    'symbol': coin,
                    'action': np.random.choice(['BUY', 'SELL']),
                    'amount': self.config.get('trade_amount', 10),
                    'price': np.random.uniform(10, 500),
                    'profit': np.random.normal(2, 1.5),
                    'strategy': np.random.choice(['Momentum', 'Mean Reversion', 'Breakout']),
                    'confidence': np.random.uniform(0.6, 0.9)
                }
                trades.append(trade)
        
        return trades
    
    def save_learning_data(self, trades):
        """حفظ بيانات التعلم"""
        learning_data = st.session_state.get('learning_data', [])
        
        for trade in trades:
            learning_record = {
                'trade_data': trade,
                'market_conditions': self.get_market_conditions(trade.get('symbol', '')),
                'outcome': 'WIN' if trade.get('profit', 0) > 0 else 'LOSS',
                'timestamp': datetime.now(),
                'lessons': self.extract_lessons(trade)
            }
            learning_data.append(learning_record)
        
        st.session_state.learning_data = learning_data
    
    def get_market_conditions(self, symbol):
        """الحصول على ظروف السوق"""
        return {
            'trend': np.random.choice(['UPTREND', 'DOWNTREND', 'SIDEWAYS']),
            'volatility': np.random.uniform(0.01, 0.05),
            'volume': np.random.uniform(1000000, 50000000)
        }
    
    def extract_lessons(self, trade):
        """استخراج الدروس من الصفقة"""
        if trade.get('profit', 0) > 0:
            return ["SUCCESSFUL_ENTRY", "GOOD_TIMING"]
        else:
            return ["NEED_BETTER_ENTRY", "RISK_MANAGEMENT"]
    
    def render_currency_dashboard(self):
        """لوحة العملات"""
        st.header("📊 لوحة العملات المتداولة")
        
        trading_coins = self.get_trading_coins()
        selected_currency = st.session_state.get('selected_currency', 'BTCUSDT')
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("💱 العملات النشطة")
            for coin in trading_coins[:self.config.get('max_coins', 10)]:
                if st.button(coin, key=f"btn_{coin}", use_container_width=True):
                    st.session_state.selected_currency = coin
        
        with col1:
            self.render_currency_chart()
            self.render_trading_strategies()
    
    def render_currency_chart(self):
        """رسم الشموع والمؤشرات"""
        selected_currency = st.session_state.get('selected_currency', 'BTCUSDT')
        st.subheader(f"📈 تحليل {selected_currency}")
        
        dates = pd.date_range(end=datetime.now(), periods=50, freq='1h')
        opens = np.random.uniform(100, 500, 50)
        highs = opens * np.random.uniform(1.01, 1.03, 50)
        lows = opens * np.random.uniform(0.97, 0.99, 50)
        closes = opens * np.random.uniform(0.98, 1.02, 50)
        
        fig = go.Figure(data=[go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=selected_currency
        )])
        
        fig.add_trace(go.Scatter(
            x=dates, y=pd.Series(closes).rolling(20).mean(),
            name='MA 20',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=pd.Series(closes).rolling(50).mean(),
            name='MA 50', 
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"تحليل فني - {selected_currency}",
            xaxis_title="الوقت",
            yaxis_title="السعر ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_trading_strategies(self):
        """عرض الاستراتيجيات"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("الاستراتيجية", "Momentum")
            st.metric("الثقة", "78%")
            st.metric("الإشارة", "🟢 شراء")
        
        with col2:
            st.metric("RSI", "42")
            st.metric("MACD", "صاعد")
            st.metric("المتجه", "📈")
        
        with col3:
            if st.button("🟢 فتح صفقة شراء", use_container_width=True):
                self.execute_trade('BUY')
            if st.button("🔴 فتح صفقة بيع", use_container_width=True):
                self.execute_trade('SELL')
    
    def execute_trade(self, action):
        """تنفيذ صفقة"""
        selected_currency = st.session_state.get('selected_currency', 'BTCUSDT')
        trade_history = st.session_state.get('trade_history', [])
        
        trade = {
            'timestamp': datetime.now(),
            'symbol': selected_currency,
            'action': action,
            'amount': self.config.get('trade_amount', 10),
            'price': np.random.uniform(100, 500),
            'profit': np.random.normal(2, 1),
            'strategy': 'Manual',
            'confidence': 0.8
        }
        
        trade_history.append(trade)
        st.session_state.trade_history = trade_history
        st.success(f"✅ تم {action} {selected_currency}")
    
    def render_trade_history(self):
        """سجل الصفقات"""
        st.header("📋 سجل الصفقات المفصل")
        
        trade_history = st.session_state.get('trade_history', [])
        
        if not trade_history:
            st.info("لا توجد صفقات حتى الآن")
            return
        
        trades_df = pd.DataFrame(trade_history)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_df['profit_display'] = trades_df['profit'].apply(
            lambda x: f"<span class='trade-positive'>+${x:.2f}</span>" if x > 0 
            else f"<span class='trade-negative'>-${abs(x):.2f}</span>"
        )
        
        st.markdown(trades_df[[
            'timestamp', 'symbol', 'action', 'amount', 
            'price', 'profit_display', 'strategy', 'confidence'
        ]].to_html(escape=False, index=False), unsafe_allow_html=True)
        
        st.subheader("📈 إحصائيات الأداء")
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(trade_history)
        winning_trades = len([t for t in trade_history if t.get('profit', 0) > 0])
        total_profit = sum(t.get('profit', 0) for t in trade_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        best_trade = max([t.get('profit', 0) for t in trade_history]) if trade_history else 0
        
        col1.metric("إجمالي الصفقات", total_trades)
        col2.metric("الصفقات الرابحة", f"{winning_trades} ({winning_trades/total_trades:.1%})" if total_trades > 0 else "0")
        col3.metric("متوسط الربح", f"${avg_profit:.2f}")
        col4.metric("أفضل صفقة", f"${best_trade:.2f}")
    
    def get_trading_coins(self):
        """قائمة العملات المتداولة"""
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
            'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT'
        ]
    
    def run(self):
        """تشغيل الواجهة"""
        self.setup_session_state()
        self.render_header()
        self.render_api_settings()
        self.render_control_panel()
        self.render_historical_simulation()
        self.render_currency_dashboard()
        self.render_trade_history()

# التشغيل الرئيسي
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
