# Note: You can use any Debian/Ubuntu based image you want.
# FROM 444488357543.dkr.ecr.ap-southeast-2.amazonaws.com/easi-workflows-snap9:test
# Use this to test a remote image from ECR, usually helps in debugging workflow sections
# FROM 444488357543.dkr.ecr.ap-southeast-2.amazonaws.com/easi-workflows-acolite:develop.latest
# Use this to test a local image from Local Docker registry or a provisional image on ECR
# helps with debugging the image itself and acolite
FROM local/easi-workflows-acolite:test
# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Install coverage analysis and asyncio, temp until enabled in base container
RUN apt-get update && apt-get install -y git-lfs
RUN /env/bin/pip install pytest-asyncio pytest-cov
# Create a non-root user to use if not already available - see https://aka.ms/vscode-remote/containers/non-root-user.
RUN if [ $(getent passwd $USERNAME) ]; then \
        # If exists, see if we need to tweak the GID/UID
        if [ "$USER_GID" != "1000" ] || [ "$USER_UID" != "1000" ]; then \
            groupmod --gid $USER_GID $USERNAME \
            && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
            && chown -R $USER_UID:$USER_GID /home/$USERNAME; \
        fi; \
    else \
        # Otherwise create the non-root user
        groupadd --gid $USER_GID $USERNAME \
        && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
        && usermod -a -G users $USERNAME \
        # Add sudo support for the non-root user
        && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
        && chmod 0440 /etc/sudoers.d/$USERNAME; \
    fi

ENTRYPOINT ["/bin/tini", "-s", "--", "with_bootstrap"]
CMD ["/bin/bash"]