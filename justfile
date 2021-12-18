version := `python3 -c "from configparser import ConfigParser; p = ConfigParser(); p.read('setup.cfg'); print(p['metadata']['version'])"`

default:
	@echo "\"just publish\"?"

tag:
	@if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	curl -H "Authorization: token `cat ~/.github-access-token`" -d '{"tag_name": "v{{version}}"}' https://api.github.com/repos/nschloe/pacopy/releases

upload: clean
	@if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	# https://stackoverflow.com/a/58756491/353337
	python3 -m build --sdist --wheel .
	twine upload dist/*

publish: tag upload

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
	@rm -rf src/*.egg-info/ build/ dist/ .tox/

format:
	isort .
	black .
	blacken-docs README.md

lint:
	black --check .
	flake8 .

mp4:
	ffmpeg -framerate 4 -i fig%03d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
