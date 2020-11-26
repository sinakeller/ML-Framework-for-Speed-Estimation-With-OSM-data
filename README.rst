ML-Framework-for-Speed-Estimation-With-OSM-data
===============================================

This repository consists of a machine learning framework that can be applied to estimate the average speed of road segments in rural regions based on OpenStreetMap road network data. The detailed description of the methodology is given in the journal paper "Machine Learning Framework for the Estimation of Average Speed in Road Networks with OpenStreetMap Data", which can be found `here <https://www.mdpi.com/2220-9964/9/11/638>`_
.
 
The repository further contains two example applications of the framework, each in a separate jupyter notebook. The `nnsw_example.ipynb <nnsw_example.ipynb>`_ notebook shows the application of the framework on the NNSW ( Australia) dataset without additionally generated features. The `bm_example.ipynb <bm_example.ipynb>`_ offers the application to the biomaule dataset with the additional use of the SOM-generated features.

**Note that this repository's applied target values differ from the actual average speed values due to copyright reasons.** The reference values in this repository are generated with the `Fuzzy-Framework <https://github.com/johannaguth/Fuzzy-Framework-for-Speed-Estimation#fuzzy-framework-for-speed-estimation>`_ and are therefore much more comfortable to predict.

.. ToDos: Include citation, update text.


Description
-----------

:License:
    `[CC BY 4.0] <LICENSE>`_

:Authors:
 .. line-block::
   `Raoul Gabriel <mailto:r.gabriel@ci-tec.de>`_, `ci-Tec GmbH <https://www.ci-tec.de>`_
   `Sina Keller <mailto:sina.keller@kit.edu>`_, `Karlsruhe Institute of Technology, Institute of Photogrammetry and Remote Sensing <https://ipf.kit.edu>`_

:Citation:
    see `Citation`_

:Paper:
    In review.

:Requirements:
    Python 3 with these `packages <requirements.txt>`_





Citation
--------

**Paper:**
.. code:: bibtex

    @article{keller2020machine,
        title={Machine Learning Framework for the Estimation of Average Speed in Rural Road Networks with OpenStreetMap Data},
        author={Keller, Sina and Gabriel, Raoul and Guth, Johanna},
        journal={ISPRS International Journal of Geo-Information},
        volume={9},
        number={11},
        pages={638},
        year={2020},
        publisher={Multidisciplinary Digital Publishing Institute}
    }


**Code:**

Raoul Gabriel and Sina Keller, "Machine Learning Framework for Speed Estimation of Roads With OpenStreetMap Data", Zenodo, `10.5281/zenodo.4012277 <http://doi.org/10.5281/zenodo.4012277>`_, 2020.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4012277.svg
    :target:  https://doi.org/10.5281/zenodo.4012277
    :alt: DOI

.. code:: bibtex

    @misc{gabriel_keller2020speed_estimation,
        author = {Gabriel, Raoul and Keller, Sina},
        title = {{Machine Learning Framework for Speed Estimation of Roads With OpenStreetMap Data}},
        year = {2020},
        DOI = {10.5281/zenodo.4012277},
        publisher = {Zenodo},
        howpublished = {\href{https://doi.org/10.5281/zenodo.4012277}{doi.org/10.5281/zenodo.4012277}}
    }
