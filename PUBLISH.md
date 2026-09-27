# Publishing checklist (deadline: 27 Sep 2026, midnight)

Run these from `/data/wiut_hackathon` in a terminal. Each step is one or two commands.

## 1. Push the website commit and make the repo public

```bash
git push origin main
gh repo edit Winner-121/Destroyerrrrs --visibility public --accept-visibility-change-consequences
```

## 2. Turn on GitHub Pages (serves `docs/` as the team site)

```bash
gh api -X POST repos/Winner-121/Destroyerrrrs/pages -f build_type=legacy -f 'source[branch]=main' -f 'source[path]=/docs'
```

The site appears within ~2 minutes at **https://winner-121.github.io/Destroyerrrrs/**
(If the API call says Pages already exists, use `-X PUT` instead of `-X POST`.)

## 3. Create the Hugging Face Space for the live demo

1. Sign up / log in at https://huggingface.co, then create a token at https://huggingface.co/settings/tokens (type: write).
2. Create a Space: https://huggingface.co/new-space — name `destroyerrrrs-traffic-events`, SDK **Gradio**, hardware **CPU basic (free)**, public.
3. Push the prepared folder (everything is in `work/space/`):

```bash
pip install -U huggingface_hub   # once
huggingface-cli login            # paste the token
cd work/space
git init && git lfs install
git remote add origin https://huggingface.co/spaces/<YOUR_HF_USERNAME>/destroyerrrrs-traffic-events
git add -A && git commit -m "traffic event demo" && git push -u origin main --force
```

The Space builds for ~5 minutes, then runs at
`https://<YOUR_HF_USERNAME>-destroyerrrrs-traffic-events.hf.space` (also shown on the Space page).

## 4. Put the demo URL into the website

```bash
cd /data/wiut_hackathon
sed -i 's#DEMO_URL_PLACEHOLDER#https://<YOUR_HF_USERNAME>-destroyerrrrs-traffic-events.hf.space#' docs/index.html
git commit -am "website: demo URL" && git push origin main
```

## 5. Tag the submission

```bash
git tag -a submission -m "Elimination submission" && git push origin submission
```

Submit: repository link `https://github.com/Winner-121/Destroyerrrrs` (tag `submission`) and
website link `https://winner-121.github.io/Destroyerrrrs/`.
