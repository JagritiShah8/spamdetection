"""Custom styling and CSS tokens for Dark Navy & Cyan Streamlit Theme."""

DARK_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Font & Canvas */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 15% 15%, #111a33 0%, #0b0f19 55%, #070a12 100%);
    color: #f1f5f9;
}

/* Header styling */
.hero-header {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #38bdf8 0%, #60a5fa 40%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

/* Polished Dark Cards */
.dark-card {
    background: rgba(17, 24, 39, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 14px;
    padding: 1.4rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(56, 189, 248, 0.05);
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.dark-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
}

.card-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metric Display Boxes */
.metric-box {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    margin-bottom: 10px;
}

.metric-title {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #38bdf8;
}

.metric-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 2px;
}

/* Badges */
.badge-spam {
    background: rgba(244, 63, 94, 0.18);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.5);
    padding: 6px 18px;
    border-radius: 24px;
    font-weight: 800;
    font-size: 1.1rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.02em;
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.25);
}

.badge-ham {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.5);
    padding: 6px 18px;
    border-radius: 24px;
    font-weight: 800;
    font-size: 1.1rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.02em;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.25);
}

.badge-uncertain {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.5);
    padding: 6px 18px;
    border-radius: 24px;
    font-weight: 800;
    font-size: 1.1rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.02em;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.25);
}

/* Chips for Words */
.chip-spam {
    display: inline-block;
    background: rgba(244, 63, 94, 0.15);
    color: #fda4af;
    border: 1px solid rgba(244, 63, 94, 0.35);
    border-radius: 16px;
    padding: 3px 10px;
    margin: 3px;
    font-size: 0.82rem;
    font-weight: 600;
}

.chip-ham {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: #86efac;
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 16px;
    padding: 3px 10px;
    margin: 3px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Highlighted Text Container */
.highlight-container {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 1.2rem;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #f8fafc;
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 29, 0.95);
    border-right: 1px solid rgba(56, 189, 248, 0.12);
}

/* Streamlit Native Tabs Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid rgba(148, 163, 184, 0.1);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    white-space: pre-wrap;
    background: transparent;
    border-radius: 8px;
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 600;
    padding: 0 16px;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
}

/* Accordion Styling */
.streamlit-expanderHeader {
    background: rgba(15, 23, 42, 0.7) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    color: #f1f5f9 !important;
}

/* Alert Boxes Customization */
div[data-testid="stAlert"] {
    background: rgba(17, 24, 39, 0.9);
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}
</style>
"""


def apply_custom_theme() -> None:
    """Inject dark navy custom CSS into the Streamlit app."""
    import streamlit as st
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
