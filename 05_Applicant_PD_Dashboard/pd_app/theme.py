"""Portfolio styling for the Streamlit app. Written by AI.

Colours, fonts and card styles copied from smallchuch.github.io/styles.css,
so the dashboard looks like part of the portfolio site. app.py calls
inject() once and then uses the small HTML helpers below for the hero and
the result panel. Everything else is plain Streamlit.
"""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --jac-100:#D7CEE2; --jac-200:#A995C0; --jac-300:#7A5C9E; --jac-400:#5C4577;
  --gold-100:#EFE1C5; --gold-400:#96742D;
  --olive-100:#D1D5CA; --olive-400:#4D563B;
  --surface:#FDFCFB; --subtle:#FAF9F5; --border:#C1C1C1;
  --text:#2F2F2F; --text-2:#787878; --text-3:#7C7B78;
  --safe-bg:#F4F9F0; --safe-text:#5B7944;
  --watch-bg:#FFFAED; --watch-text:#9D8232;
  --bad-bg:#FCEBEA; --bad-text:#8D2623;
  --head:"Inter","DM Sans",Arial,sans-serif;
  --body:"DM Sans","Inter",Arial,sans-serif;
  --mono:"JetBrains Mono","Courier New",monospace;
}
.stApp, input, textarea, [data-baseweb="select"] div, button p { font-family: var(--body); }
h1, h2, h3, h4 { font-family: var(--head) !important; letter-spacing: -0.01em; }
.block-container { max-width: 1080px; padding-top: 3rem; }
header[data-testid="stHeader"] { display: none; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

.eyebrow { font-family: var(--mono); text-transform: uppercase; letter-spacing: .14em;
           font-size: .72rem; font-weight: 500; color: var(--gold-400); margin-bottom: .4rem; }
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 .4rem 0; }
.hero h1 .accent { color: var(--jac-300); }
.lead { color: var(--text-2); font-size: 1.02rem; max-width: 680px; margin-bottom: 1.4rem; }

.result { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
          padding: 1.6rem 1.8rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 1.6rem; }
.result.empty { border-style: dashed; color: var(--text-3); text-align: center; padding: 2.2rem; }
.result-grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 2rem; align-items: center; }
.pd-label { font-family: var(--mono); font-size: .72rem; letter-spacing: .12em;
            text-transform: uppercase; color: var(--text-3); }
.pd-num { font-family: var(--head); font-size: 3.6rem; font-weight: 700;
          color: var(--jac-300); line-height: 1.05; margin: .2rem 0 .5rem; }
.pill { display: inline-block; font-family: var(--mono); font-size: .72rem; font-weight: 500;
        padding: .25rem .65rem; border-radius: 999px; }
.pill.safe  { background: var(--safe-bg);  color: var(--safe-text); }
.pill.watch { background: var(--watch-bg); color: var(--watch-text); }
.pill.bad   { background: var(--bad-bg);   color: var(--bad-text); }
.grade-row { display: flex; align-items: center; gap: .8rem; margin: .1rem 0 .6rem; }
.grade { display: inline-flex; align-items: baseline; gap: .45rem; border-radius: 10px;
         padding: .35rem .8rem; font-family: var(--head); font-weight: 700; }
.grade .g-num { font-size: 1.35rem; }
.grade .g-lbl { font-family: var(--body); font-weight: 500; font-size: .9rem; }
.grade.safe  { background: var(--safe-bg);  color: var(--safe-text); }
.grade.watch { background: var(--watch-bg); color: var(--watch-text); }
.grade.bad   { background: var(--bad-bg);   color: var(--bad-text); }
.scale { display: flex; gap: 3px; margin-top: .7rem; max-width: 320px; }
.scale span { flex: 1; height: 6px; border-radius: 3px; background: var(--subtle);
              border: 1px solid #E4E1DA; }
.scale span.on { background: var(--jac-300); border-color: var(--jac-300); }
.scale-lbl { font-family: var(--mono); font-size: .66rem; color: var(--text-3); margin-top: .25rem; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.stat .num { font-family: var(--head); font-size: 1.45rem; font-weight: 700; color: var(--text); }
.stat .lbl { font-size: .82rem; color: var(--text-2); }
.model-line { font-family: var(--mono); font-size: .72rem; color: var(--text-3);
              border-top: 1px solid var(--subtle); margin-top: 1.2rem; padding-top: .8rem; }
.model-line b { color: var(--jac-400); font-weight: 500; }
.challengers { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: .8rem; }
.chal { border: 1px solid var(--border); border-radius: 10px; padding: .5rem .8rem;
        font-family: var(--mono); font-size: .75rem; color: var(--text-2); background: var(--subtle); }
.chal b { color: var(--text); font-size: .95rem; }

[data-testid="stForm"] { background: var(--surface); border: 1px solid var(--border) !important;
                         border-radius: 14px; padding: 1.4rem 1.6rem; }
.section-kicker { font-family: var(--mono); font-size: .7rem; letter-spacing: .12em;
                  text-transform: uppercase; color: var(--jac-300); margin: .2rem 0 .6rem;
                  border-bottom: 1px solid var(--jac-100); padding-bottom: .35rem; }
.stButton button, [data-testid="stFormSubmitButton"] button {
    border-radius: 999px; padding: .55rem 1.6rem; font-weight: 600; }
/* Field labels: heading font, bolder and darker, so the question stands
   apart from the answer (inputs keep the lighter body font). */
[data-testid="stWidgetLabel"] p {
    font-family: var(--head) !important; font-weight: 600 !important;
    font-size: .84rem !important; color: var(--text) !important; letter-spacing: .005em; }
[data-testid="stExpander"] { border-radius: 10px; }
/* the only iframe on the page is the invisible scroll-to-top helper */
[data-testid="stElementContainer"]:has(> iframe), .element-container:has(> iframe) { display: none; }
[data-testid="stExpander"] summary p { font-size: .88rem; color: var(--text-2); }

@media (max-width: 720px) {
  .result-grid { grid-template-columns: 1fr; gap: 1rem; }
  .pd-num { font-size: 2.8rem; }
  .hero h1 { font-size: 1.8rem; }
}
</style>
"""


def scroll_to_top():
    """After scoring, bring the result panel into view (matters on a phone,
    where the button sits below the fold). Uses st.iframe on new Streamlit
    versions and falls back to components.html on older ones."""
    js = """<script>
      const d = window.parent.document;
      (d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main') || {}).scrollTo?.({top: 0, behavior: 'smooth'});
      window.parent.scrollTo({top: 0, behavior: 'smooth'});
    </script>"""
    if hasattr(st, "iframe"):
        st.iframe(js, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(js, height=0)


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow, title_html, lead):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
                f'<h1>{title_html}</h1><p class="lead">{lead}</p></div>',
                unsafe_allow_html=True)


def kicker(text):
    st.markdown(f'<div class="section-kicker">{text}</div>', unsafe_allow_html=True)


def empty_result(text):
    st.markdown(f'<div class="result empty">{text}</div>', unsafe_allow_html=True)


def result_panel(pd_value, pill_class, pill_text, stats, model_line, challengers_html="",
                 grade=None):
    """stats: list of (number, label). grade: dict from risk_grades.json.
    Everything is pre-formatted text."""
    grade_html = ""
    if grade:
        bars = "".join(f'<span class="{"on" if i <= grade["grade"] else ""}"></span>'
                       for i in range(1, 6))
        grade_html = (f'<div class="grade-row"><div class="grade {grade["style"]}">'
                      f'<span class="g-num">Grade {grade["grade"]}</span>'
                      f'<span class="g-lbl">{grade["label"]}</span></div></div>')
        pill_tail = (f'<div class="scale">{bars}</div>'
                     f'<div class="scale-lbl">Grade 1 lowest risk · Grade 5 highest</div>')
    else:
        pill_tail = ""
    stat_html = "".join(f'<div class="stat"><div class="num">{n}</div>'
                        f'<div class="lbl">{l}</div></div>' for n, l in stats)
    st.markdown(f"""
<div class="result">
  <div class="result-grid">
    <div>
      <div class="pd-label">Estimated probability of payment difficulty</div>
      <div class="pd-num">{pd_value}</div>
      {grade_html}
      <span class="pill {pill_class}">{pill_text}</span>
      {pill_tail}
    </div>
    <div class="stats">{stat_html}</div>
  </div>
  {challengers_html}
  <div class="model-line">{model_line}</div>
</div>""", unsafe_allow_html=True)
