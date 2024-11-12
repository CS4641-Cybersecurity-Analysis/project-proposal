# Contributing
## Resources
[Streamlit Walkthrough Notion](https://nostalgic-beauty-37e.notion.site/Streamlit-Walkthrough-2de3c47c3c4b4cbfbccb6c264663c8a0)
[Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)

## Setup
Use the conda environment defined in `environment/env_win.yml` or just ensure that at least python3 is installed with streamlit.

`conda env create -f env_win.yml` to create the environment and `conda activate cs4641-project-proposal` to activate it. Note the `yml` file detailing the environment is only tested on Windows, but will likely work on Linux/MacOS.

Edit `Proposal.py` in the `app` directory to make changes to the page. Or edit `<title>.py` in the `app/pages` directory to modify other pages.

Run `streamlit run app/main.py` or `python -m streamlit run app/main.py` while in the project root directory to view the app in the browser.
