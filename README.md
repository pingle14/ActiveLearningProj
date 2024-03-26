# ActiveLearningProj
> `query_strategies` modify the active learning strategies to serve a streaming pool


## How to run this project experiments
> `simple_linreg_exp.py` is the experiment for linear regression. Specify parameters as follows:
```
-n := num_rounds (default = 10)
-c := num_coeffs (default = 5)
-s := initial_sample_sz (default = 20)
-p := pool_sz (default = 1000)
-b := budget (default = 10)
-i := iter_per_algo (default = 10)
-l := long_experiment (default = false)
-v := verbose mode (default = false)
```
Example run command:
```
python3 simple_linreg_exp.py -n 1000 -c 2 -b 1 -p 1000
```
---

## Developer's Corner:

### Contributing Summary
- To format code, run `black .`
- To lint code, run `flake8 .`
- To install requirements, run `pip install -r requirements.txt`

### Contributing

1. Fork this Repo
2. Clone the Repo onto your computer -- You may need to setup an SSH Key on your device.
 - Run `pip install -r requirements.txt` to get all the packages you need.
3. Create a branch (`git checkout -b new-feature`)
4. Make Changes
5. Run necessary quality assurance tools
 - [Formatter](#Formatter), [Linter](#Linter), and Test your code.
6. Add your changes (`git commit -am "Commit Message"` or `git add <whatever files/folders you want to add>` followed by `git commit -m "Commit Message"`)
7. Push your changes to the repo (`git push origin new-feature`)
8. Create a pull request

---
## Code Quality Tools
### [black formatter](https://github.com/psf/black) automatically formats code

1. Run `pip install black` to get the package.
2. After making changes, run `black ./`.

### [flake8](https://github.com/pycqa/flake8) lints code
> Notifies you if any code is not currently up to Python standards.

1. Run `pip install flake8` to get the package.
2. After making changes, run `flake8`.


---