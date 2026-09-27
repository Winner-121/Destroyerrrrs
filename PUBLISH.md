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

## 3. Live demo (done): Modal, free CPU tier

Deployed from this repo with `modal deploy demo/modal_app.py` (account amir-kuldashev):
https://amir-kuldashev--destroyerrrrs-traffic-events-web.modal.run
It scales to zero when idle (first request after idle takes ~30 s) and costs nothing within
the $30/month free credit. Redeploy with the same command after changing `demo/app.py` or `src/`.

## 4. Push the site update with the demo URL

```bash
git push origin main
```

## 5. Tag the submission

```bash
git tag -a submission -m "Elimination submission" && git push origin submission
```

Submit: repository link `https://github.com/Winner-121/Destroyerrrrs` (tag `submission`), website `https://winner-121.github.io/Destroyerrrrs/`, live demo `https://amir-kuldashev--destroyerrrrs-traffic-events-web.modal.run` (deployed with `modal deploy demo/modal_app.py`).
