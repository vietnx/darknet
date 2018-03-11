GPU=1
CUDNN=1
OPENCV=1
OPENMP=0
DEBUG=0

ARCH= -gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

PROJECT := darknet
BUILD_DIR := build
#PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib

LIBRARY_NAME := $(PROJECT)
LIB_BUILD_DIR := $(BUILD_DIR)/lib
STATIC_NAME_SHORT := lib$(LIBRARY_NAME).a
STATIC_NAME := $(LIB_BUILD_DIR)/$(STATIC_NAME_SHORT)
DYNAMIC_VERSION_MAJOR 		:= 1
DYNAMIC_VERSION_MINOR 		:= 0
DYNAMIC_VERSION_REVISION 	:= 0
DYNAMIC_NAME_SHORT := lib$(LIBRARY_NAME).so
DYNAMIC_SONAME_SHORT := $(DYNAMIC_NAME_SHORT).$(DYNAMIC_VERSION_MAJOR).$(DYNAMIC_VERSION_MINOR)
DYNAMIC_VERSIONED_NAME_SHORT := $(DYNAMIC_SONAME_SHORT).$(DYNAMIC_VERSION_REVISION)
DYNAMIC_NAME := $(LIB_BUILD_DIR)/$(DYNAMIC_VERSIONED_NAME_SHORT)
LABELS_PATH := $(PREFIX)/share/$(PROJECT)/labels

VERSIONFLAGS += -Wl,-soname,$(DYNAMIC_SONAME_SHORT)

VPATH=./src/:./examples
SLIB=$(DYNAMIC_NAME)
ALIB=$(STATIC_NAME)
EXEC=$(PROJECT)
OBJDIR=$(BUILD_DIR)/obj/

CC ?= gcc
CXX ?= g++
NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-ccbin=$(CXX)
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -Wno-unused-result -fPIC

ifneq ($(PREFIX),)
CFLAGS += -DDARKNET_LABELS_PATH=\"$(LABELS_PATH)\"
endif

ifeq ($(OPENMP), 1)
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1)
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv`
COMMON+= `pkg-config --cflags opencv`
endif

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1)
COMMON+= -DCUDNN
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o attention.o darknet.o demo.o visualization.o
ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj lib $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(VERSIONFLAGS) -shared
	@ cd $(BUILD_DIR)/lib; rm -f $(DYNAMIC_SONAME_SHORT); ln -s $(DYNAMIC_VERSIONED_NAME_SHORT) $(DYNAMIC_SONAME_SHORT)
	@ cd $(BUILD_DIR)/lib; rm -f $(DYNAMIC_NAME_SHORT);   ln -s $(DYNAMIC_SONAME_SHORT) $(DYNAMIC_NAME_SHORT)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p $(OBJDIR)
lib:
	mkdir -p $(LIB_BUILD_DIR)
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: install
install: all
#install include file
	install -d $(DESTDIR)$(PREFIX)/include/$(PROJECT)/
	install -m 644 include/darknet.h $(DESTDIR)$(PREFIX)/include/$(PROJECT)/
#install library files
	install -d $(DESTDIR)$(LIBDIR)
	install -m 644 $(STATIC_NAME) $(DESTDIR)$(LIBDIR)
	install -m 644 $(DYNAMIC_NAME) $(DESTDIR)$(LIBDIR)
	cd $(DESTDIR)$(LIBDIR); rm -f $(DYNAMIC_SONAME_SHORT); ln -s $(DYNAMIC_VERSIONED_NAME_SHORT) $(DYNAMIC_SONAME_SHORT)
	cd $(DESTDIR)$(LIBDIR); rm -f $(DYNAMIC_NAME_SHORT); ln -s $(DYNAMIC_SONAME_SHORT) $(DYNAMIC_NAME_SHORT)
#install executable files
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(PROJECT) $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(LABELS_PATH)
	cp data/labels/* $(DESTDIR)$(LABELS_PATH)

.PHONY: uninstall
uninstall:
	# remove include files
	rm -rf $(DESTDIR)$(PREFIX)/include/$(PROJECT)
	# remove libraries
	rm -f $(DESTDIR)$(LIBDIR)/$(STATIC_NAME_SHORT)
	rm -f $(DESTDIR)$(LIBDIR)/$(DYNAMIC_NAME_SHORT)
	rm -f $(DESTDIR)$(LIBDIR)/$(DYNAMIC_SONAME_SHORT)
	rm -f $(DESTDIR)$(LIBDIR)/$(DYNAMIC_VERSIONED_NAME_SHORT)
	# remove executable files
	rm -f $(DESTDIR)$(PREFIX)/bin/$(PROJECT)
	rm -rf $(DESTDIR)$(PREFIX)/share/$(PROJECT)
	# remove empty folders
	-rmdir $(DESTDIR)$(PREFIX)/include
	-rmdir $(DESTDIR)$(LIBDIR)
	-rmdir $(DESTDIR)$(PREFIX)/bin
	-rmdir $(DESTDIR)$(PREFIX)/share

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) $(EXEC)

