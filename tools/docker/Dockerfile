FROM gcr.io/tensorflow/tensorflow:1.3.0-gpu-py3
# nvidia-docker run -d --restart=always -p 3022:22 -p 36006:6006 -p 38080:8080 --name stayrascal-host -v /data:/data stayrascal/tensorflow

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN mkdir /root/.ssh && echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDKuzjdgRYf6Svj0SIyt3l2DGRsEg7LXqqnK7xobWpDunp4fkJNU944qlC2dRRUcYWa4VwHpHGRw5EEfbVkiHWvewQeD5vS0c0WOt2dSRSE5ytfXgh1JJIjDb0Xyyc+/no1cFz6LP1sKN/P4YksdFaToqmn5mVNcD5t+Rtr1v38uKmxk/R2P/G7G/+yRQaEuFAoWqqZN2Lev8sT7Icbfvvs/ESvgEVlAPq9ev0eVI8A/X35s4G7bNxm3dUCtIspeoj8GcRri8NVs1ttKUVIqCbc2Ka/cntSAfhcSoOvvHRvnwbgx/Qt6JaRSTcNMbGv57YoKJqAgN7H8DPvbI7HOpUR zpwu@thoughtworks.com' > /root/.ssh/authorized_keys && chmod 644 /root/.ssh/authorized_keys

RUN apt-get install -y vim git tmux axel tree nmon
RUN pip3 install scikit-learn scipy six sklearn protobuf h5py gym matplotlib numpy PyOpenGL opencv-python virtualenv
RUN echo 'export PYTHONPATH=.:$PYTHONPATH' >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/nvidia/lib64' >> /root/.bashrc

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]