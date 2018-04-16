GPU=1
CUDNN=1
OPENCV=1
CUDNN_HALF=0
OPENMP=0
DEBUG=0

ARCH= -gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

# GP100/Tesla P100 ֠DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson Tx1 uncomment:
# ARCH= -gencode arch=compute_51,code=[sm_51,compute_51]

# For Jetson Tx2 or Drive-PX2 uncomment:
# ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]

PROJECT := darknet
BUILD_DIR := build
#PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib

LIBRARY_NAME := $(PROJECT)
LIB_BUILD_DIR := $(BUILD_DIR)/lib
STATIC_NAME_SHORT := lib$(LIBRARY_NAME).a
STATIC_NAME := $(LIB_BUILD_DIR)/$(STATIC_NAME_SHORT)
DYNAMIC_VERSION_MAJOR 		:= 2
DYNAMIC_VERSION_MINOR 		:= 0
DYNAMIC_VERSION_REVISION 	:= 0
ifeq ($(OS),Windows_NT)
DYNAMIC_NAME_SHORT := lib$(LIBRARY_NAME).dll
IMPORT_NAME_SHORT := lib$(LIBRARY_NAME).lib
DYNAMIC_NAME := $(LIB_BUILD_DIR)/$(DYNAMIC_NAME_SHORT)
IMPORT_NAME := $(LIB_BUILD_DIR)/$(IMPORT_NAME_SHORT)
else
DYNAMIC_NAME_SHORT := lib$(LIBRARY_NAME).so
DYNAMIC_SONAME_SHORT := $(DYNAMIC_NAME_SHORT).$(DYNAMIC_VERSION_MAJOR).$(DYNAMIC_VERSION_MINOR)
DYNAMIC_VERSIONED_NAME_SHORT := $(DYNAMIC_SONAME_SHORT).$(DYNAMIC_VERSION_REVISION)
DYNAMIC_NAME := $(LIB_BUILD_DIR)/$(DYNAMIC_VERSIONED_NAME_SHORT)
VERSIONFLAGS += -Wl,-soname,$(DYNAMIC_SONAME_SHORT)
endif
LABELS_PATH := $(PREFIX)/share/$(PROJECT)/labels

VPATH=./src/:./examples
SLIB=$(DYNAMIC_NAME)
ALIB=$(STATIC_NAME)
EXEC=$(PROJECT)
OBJDIR=$(BUILD_DIR)/obj/
EXECOBJDIR = $(BUILD_DIR)/execobj/

ifeq ($(OS),Windows_NT)
CC := cl
CXX := cl
NVCC=nvcc
NVCCFLAGS = -ccbin=cl
CFLAGS =  -I/d/Projects/Libs/pthreads-w32-2-9-1-release/Pre-built.2/include
CFLAGS += -DWIN32 -D_WINDOWS -DNDEBUG -DHAVE_STRUCT_TIMESPEC=1 -DHAVE_SIGNAL_H=1
CFLAGS += -MD
CFLAGS_SHARED := -DDLL_EXPORT
LDFLAGS = -LIBPATH:/d/Projects/Libs/pthreads-w32-2-9-1-release/Pre-built.2/lib/x64 pthreadVC2.lib
LDFLAGS += kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib
LDFLAGS += -MACHINE:X64 -SUBSYSTEM:CONSOLE -NOLOGO
LDFLAGS_SHARED := -IMPLIB:$(IMPORT_NAME)
OPTS = -O2
OBJ_EXT = .obj
else
CC ?= gcc
CXX ?= g++
NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS = -ccbin=$(CXX)
CFLAGS = -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC
LDFLAGS += -lm -pthread
OPTS = -Ofast
OBJ_EXT = .o
LDFLAGS_EXEC = -L$(LIB_BUILD_DIR) -l$(LIBRARY_NAME)
LDFLAGS_EXEC += -Wl,-rpath,$(LIB_BUILD_DIR)
endif
AR=ar
ARFLAGS=rcs
COMMON= -Iinclude/ -Isrc/

ifneq ($(PREFIX),)
CFLAGS += -DDARKNET_LABELS_PATH=\"$(LABELS_PATH)\"
endif

ifeq ($(OPENMP), 1)
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1)
OPTS = -O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
ifeq ($(OS),Windows_NT)
COMMON += -I/c/opencv/build/include
LDFLAGS += -LIBPATH:/c/opencv/build/x64/vc14/lib opencv_world331.lib
else
LDFLAGS+= `pkg-config --libs opencv`
COMMON+= `pkg-config --cflags opencv`
endif
endif

ifeq ($(GPU), 1)
COMMON += -DGPU
CFLAGS += -DGPU
ifeq ($(OS),Windows_NT)
COMMON += -I"$(CUDA_PATH)/include/"
LDFLAGS += -LIBPATH:"$(CUDA_PATH)/lib/x64"
LDFLAGS += cuda.lib cudart.lib cublas.lib curand.lib
else
COMMON += -I/usr/local/cuda/include/
LDFLAGS += -L/usr/local/cuda/lib64
LDFLAGS += -lcuda -lcudart -lcublas -lcurand
LDFLAGS+= -lstdc++
endif
endif

ifeq ($(CUDNN), 1)
COMMON+= -DCUDNN
CFLAGS+= -DCUDNN
ifeq ($(OS),Windows_NT)
COMMON += -I/d/Projects/Libs/cudnn-9.2-windows10-x64-v7.1/cuda/include
LDFLAGS += -LIBPATH:/d/Projects/Libs/cudnn-9.2-windows10-x64-v7.1/cuda/lib/x64 cudnn.lib
else
LDFLAGS+= -lcudnn
endif

ifeq ($(CUDNN_HALF), 1)
COMMON+= -DCUDNN_HALF
CFLAGS+= -DCUDNN_HALF
endif

OBJ=gemm$(OBJ_EXT) utils$(OBJ_EXT) cuda$(OBJ_EXT) deconvolutional_layer$(OBJ_EXT) convolutional_layer$(OBJ_EXT) list$(OBJ_EXT) image$(OBJ_EXT) activations$(OBJ_EXT) im2col$(OBJ_EXT) col2im$(OBJ_EXT) blas$(OBJ_EXT) crop_layer$(OBJ_EXT) dropout_layer$(OBJ_EXT) maxpool_layer$(OBJ_EXT) softmax_layer$(OBJ_EXT) data$(OBJ_EXT) matrix$(OBJ_EXT) network$(OBJ_EXT) connected_layer$(OBJ_EXT) cost_layer$(OBJ_EXT) parser$(OBJ_EXT) option_list$(OBJ_EXT) detection_layer$(OBJ_EXT) route_layer$(OBJ_EXT) box$(OBJ_EXT) normalization_layer$(OBJ_EXT) avgpool_layer$(OBJ_EXT) layer$(OBJ_EXT) local_layer$(OBJ_EXT) shortcut_layer$(OBJ_EXT) activation_layer$(OBJ_EXT) rnn_layer$(OBJ_EXT) gru_layer$(OBJ_EXT) crnn_layer$(OBJ_EXT) batchnorm_layer$(OBJ_EXT) region_layer$(OBJ_EXT) reorg_layer$(OBJ_EXT) tree$(OBJ_EXT)  lstm_layer$(OBJ_EXT) l2norm_layer$(OBJ_EXT) logistic_layer$(OBJ_EXT) upsample_layer$(OBJ_EXT) yolo_layer$(OBJ_EXT)
EXECOBJA=captcha$(OBJ_EXT) lsd$(OBJ_EXT) super$(OBJ_EXT) art$(OBJ_EXT) tag$(OBJ_EXT) cifar$(OBJ_EXT) go$(OBJ_EXT) rnn$(OBJ_EXT) segmenter$(OBJ_EXT) regressor$(OBJ_EXT) classifier$(OBJ_EXT) coco$(OBJ_EXT) yolo$(OBJ_EXT) detector$(OBJ_EXT) nightmare$(OBJ_EXT) attention$(OBJ_EXT) darknet$(OBJ_EXT) demo$(OBJ_EXT) visualization$(OBJ_EXT)
ifeq ($(GPU), 1)
OBJ+=convolutional_kernels$(OBJ_EXT) deconvolutional_kernels$(OBJ_EXT) activation_kernels$(OBJ_EXT) im2col_kernels$(OBJ_EXT) col2im_kernels$(OBJ_EXT) blas_kernels$(OBJ_EXT) crop_layer_kernels$(OBJ_EXT) dropout_layer_kernels$(OBJ_EXT) maxpool_layer_kernels$(OBJ_EXT) avgpool_layer_kernels$(OBJ_EXT)
endif

EXECOBJ = $(addprefix $(EXECOBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj execobj lib $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(SLIB)
ifeq ($(OS),Windows_NT)
	$(CC) $(COMMON) $(CFLAGS) $(EXECOBJ) $(IMPORT_NAME) -Fe$@ -link $(LDFLAGS)
else
	$(CC) $(COMMON) $(CFLAGS) $(EXECOBJ) -o $@ $(LDFLAGS) $(LDFLAGS_EXEC)
endif

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
ifeq ($(OS),Windows_NT)
	$(CC) $(CFLAGS) $^ -Fe$@ -LD -link $(LDFLAGS) $(LDFLAGS_SHARED)
else
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(VERSIONFLAGS) -shared
	@ cd $(LIB_BUILD_DIR); rm -f $(DYNAMIC_SONAME_SHORT); ln -s $(DYNAMIC_VERSIONED_NAME_SHORT) $(DYNAMIC_SONAME_SHORT)
	@ cd $(LIB_BUILD_DIR); rm -f $(DYNAMIC_NAME_SHORT);   ln -s $(DYNAMIC_SONAME_SHORT) $(DYNAMIC_NAME_SHORT)
endif

$(OBJDIR)%$(OBJ_EXT): %.c $(DEPS)
ifeq ($(OS),Windows_NT)
	$(CC) $(COMMON) $(CFLAGS) $(CFLAGS_SHARED) -c $< -Fo$@
else
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@
endif

$(EXECOBJDIR)%$(OBJ_EXT): %.c $(DEPS)
ifeq ($(OS),Windows_NT)
	$(CC) $(COMMON) $(CFLAGS) -c $< -Fo$@
else
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@
endif

$(OBJDIR)%$(OBJ_EXT): %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS) $(CFLAGS_SHARED)" -c $< -o $@

obj:
	mkdir -p $(OBJDIR)
execobj:
	mkdir -p $(EXECOBJDIR)
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
ifeq ($(OS),Windows_NT)
	install -m 644 $(IMPORT_NAME) $(DESTDIR)$(LIBDIR)
else
	install -m 644 $(DYNAMIC_NAME) $(DESTDIR)$(LIBDIR)
	cd $(DESTDIR)$(LIBDIR); rm -f $(DYNAMIC_SONAME_SHORT); ln -s $(DYNAMIC_VERSIONED_NAME_SHORT) $(DYNAMIC_SONAME_SHORT)
	cd $(DESTDIR)$(LIBDIR); rm -f $(DYNAMIC_NAME_SHORT); ln -s $(DYNAMIC_SONAME_SHORT) $(DYNAMIC_NAME_SHORT)
endif
#install executable files
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(PROJECT) $(DESTDIR)$(PREFIX)/bin
ifeq ($(OS),Windows_NT)
	install -m 644 $(DYNAMIC_NAME) $(DESTDIR)$(PREFIX)/bin
endif
	install -d $(DESTDIR)$(LABELS_PATH)
	cp data/labels/* $(DESTDIR)$(LABELS_PATH)

.PHONY: uninstall
uninstall:
	# remove include files
	rm -rf $(DESTDIR)$(PREFIX)/include/$(PROJECT)
	# remove libraries
	rm -f $(DESTDIR)$(LIBDIR)/$(STATIC_NAME_SHORT)
ifeq ($(OS),Windows_NT)
	rm -f $(DESTDIR)$(LIBDIR)/$(IMPORT_NAME_SHORT)
else
	rm -f $(DESTDIR)$(LIBDIR)/$(DYNAMIC_NAME_SHORT)
	rm -f $(DESTDIR)$(LIBDIR)/$(DYNAMIC_SONAME_SHORT)
	rm -f $(DESTDIR)$(LIBDIR)/$(DYNAMIC_VERSIONED_NAME_SHORT)
endif
	# remove executable files
	rm -f $(DESTDIR)$(PREFIX)/bin/$(PROJECT)
ifeq ($(OS),Windows_NT)
	rm -f $(DESTDIR)$(PREFIX)/bin/$(DYNAMIC_NAME_SHORT)
endif
	rm -rf $(DESTDIR)$(PREFIX)/share/$(PROJECT)
	# remove empty folders
	-rmdir $(DESTDIR)$(PREFIX)/include
	-rmdir $(DESTDIR)$(LIBDIR)
	-rmdir $(DESTDIR)$(PREFIX)/bin
	-rmdir $(DESTDIR)$(PREFIX)/share
	-rmdir $(DESTDIR)$(PREFIX)

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) $(EXEC)
ifeq ($(OS),Windows_NT)
	rm -rf $(PROJECT).lib $(PROJECT).exp
endif
