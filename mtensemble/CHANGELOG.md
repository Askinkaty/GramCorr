# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres (loosely[^looseley_semver]) to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

[^looseley_semver]: We will likely only need MAJOR and MINOR version for this project.

## WIP
### Added
- evaluation script
- prediction script


## 2018-12-10
### Added
- Makefile

### Changed
- make-ified scripts for data processing
- .txt -> .cvres


## 2018-12-07
### Added
- per-error normalised confidence values
- per-error delta values of confidences

### Changed
- confidence value for predicted class:-1 is set to sqrt(min) (instead of
	max(conf_vals))
- use the original data's individual folds


## 2018-12-06
### Added
- Initial version


## x.y.z - yyyy-mm[-dd]
### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
