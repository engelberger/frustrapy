FROM us-docker.pkg.dev/colab-images/public/runtime
RUN apt update -y
RUN apt install wget mc r-base python3 python3-pip pymol libmagick++-dev libcurl4-openssl-dev libssl-dev libgit2-dev -y
RUN apt install libcurl4-gnutls-dev libxml2-dev -y
RUN python3 -m pip install numpy biopython leidenalg
RUN apt install software-properties-common -y
RUN wget https://salilab.org/modeller/9.25/modeller_9.25-1_amd64.deb
RUN env KEY_MODELLER=XXX dpkg -i modeller_9.25-1_amd64.deb
RUN Rscript -e "install.packages('ggrepel')"
RUN Rscript -e "install.packages('igraph')"
RUN Rscript -e "install.packages('dplyr')"
RUN Rscript -e "install.packages('FactoMineR')"
RUN Rscript -e "install.packages('Hmisc')"`
RUN Rscript -e "install.packages('argparse')"
RUN Rscript -e "install.packages('leiden')"
RUN Rscript -e "install.packages('magick')"
RUN Rscript -e "options(timeout=9999999)"
RUN Rscript -e "devtools::install_github('proteinphysiologylab/frustratometeR')"