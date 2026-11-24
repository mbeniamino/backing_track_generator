Forgejo runs the workflows in `.github/workflows/` when this folder doesn't exist.
Those need Windows runners, which this Forgejo instance doesn't have, so this
folder exists (with no workflows) to keep them on GitHub only.
