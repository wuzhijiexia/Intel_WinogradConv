# ======================================
# || Author: WuZheng from USTC-ACSA   ||
# || Email : zhengwu@mail.ustc.edu.cn ||
# || Data  : 2019-02-25               ||
# ======================================

# 指定编译链接的command和options
CC = icc
CFLAGS += -O3 -qopenmp -xhost -restrict -Wall -fPIC #-qopt-report=5
LD = icc
LDFLAGS += 

# release or debug
# CFLAGS += -g

# 依赖文件及相关路径
INC_DIR += $(HOME)/Intel_WinogradConv/include/
INC_DIR += $(HOME)/person/intel_2019/mkl/include/
LIB_DIR += $(HOME)/person/intel_2019/mkl/lib/intel64_lin/
LIB_DIR += $(HOME)/person/intel_2019/compilers_and_libraries_2019.2.187/linux/compiler/lib/intel64_lin/
LIB_DIR += $(BUILD_DIR)/lib/

CFLAGS += $(INC_DIR:%=-I%)

LDFLAGS += $(LIB_DIR:%=-L%)
LDFLAGS += $(LIB_DIR:%=-Wl,-rpath=%) # 通过-Wl,-rpath=, 使得execute记住链接库的路径
LDFLAGS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# 源文件相关路径
INIT_DIR = .
SRC_DIR = $(INIT_DIR)/src
TOOL_DIR = $(INIT_DIR)/tool
BUILD_DIR = $(INIT_DIR)/build

# 中间文件相关路径
SRC_OBJ_DIR = $(BUILD_DIR)/src
TOOL_OBJ_DIR = $(BUILD_DIR)/tool
SRC_OBJ_F = $(patsubst %.cpp,$(SRC_OBJ_DIR)/%.o,$(notdir $(wildcard $(SRC_DIR)/*.cpp)))
TOOL_OBJ_F = $(patsubst %.cpp,$(TOOL_OBJ_DIR)/%.o,$(notdir $(wildcard $(TOOL_DIR)/*.cpp)))

# 可执行文件相关路径
TOOL_EXE_DIR = $(BUILD_DIR)/tool
TOOL_EXE_F = $(patsubst %.cpp,$(TOOL_EXE_DIR)/%,$(notdir $(wildcard $(TOOL_DIR)/*.cpp)))

# 生成库
LIB_NAME = intel_winoconv
LIB_F = $(BUILD_DIR)/lib/lib$(LIB_NAME).so
LIBFLAGS = -shared -Wl,-soname

# 颜色
RED 		= "\e[38;5;9m"
GREEN 		= "\e[38;5;10m"
YELLOW 		= "\e[38;5;11m"
BLUE 		= "\e[38;5;12m"
PURPLE 		= "\e[38;5;13m"
WHITE 		= "\e[38;5;15m"

.PHONY: all
all: env_color $(LIB_F) $(TOOL_EXE_F)
	@echo -e $(GREEN)"-------------------- Make All Success --------------------"

$(TOOL_EXE_F): % : %.o $(LIB_F)
	$(LD) $(CFLAGS) -o $@ $^ $(LDFLAGS) -l$(LIB_NAME)

#%.o: %.cpp
#	$(CC) $(CFLAGS) -o $@ -c $<
$(SRC_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -o $@ -c $<
$(TOOL_OBJ_DIR)/%.o: $(TOOL_DIR)/%.cpp
	$(CC) $(CFLAGS) -o $@ -c $<

$(LIB_F): $(SRC_OBJ_F)
	@echo -e $(RED)"Create dynamic library, please waiting ....."
	$(CC) $(CFLAGS) $(LIBFLAGS),$(notdir $(LIB_F)) -o $@ $^ $(LDFLAGS)

.PHONY: env_color
env_color:
	@echo -e $(RED)"Begin making, please waiting ....."$(WHITE)

.PHONY: clean
clean: env_color
	@echo -e $(RED)"Clean unused files, please waiting ....."$(WHITE)
	rm -f $(SRC_OBJ_DIR)/* $(TOOL_OBJ_DIR)/*
	rm -f $(TOOL_EXE_DIR)/*
	rm -f $(LIB_F)
	@echo -e $(GREEN)"-------------------- Make Clean Success --------------------"

.PHONY: show

show: env_color
	@echo -e $(GREEN)"Show all INIT-path message:"$(WHITE)
	@echo "INIT_DIR:	$(INIT_DIR)"
	@echo "=========================================="
	
	@echo -e $(GREEN)"Show all SRC-file message:"$(WHITE)
	@echo "SRC_DIR:	$(SRC_DIR)"
	@echo "SRC_OBJ_DIR:	$(SRC_OBJ_DIR)"
	@echo "SRC_OBJ_F:	$(SRC_OBJ_F)"
	@echo "=========================================="
	
	@echo -e $(GREEN)"Show all TOOL-file message:"$(WHITE)
	@echo "TOOL_DIR:	$(TOOL_DIR)"
	@echo "TOOL_OBJ_DIR:	$(TOOL_OBJ_DIR)"
	@echo "TOOL_OBJ_F:	$(TOOL_OBJ_F)"
	@echo "TOOL_EXE_DIR:	$(TOOL_EXE_DIR)"
	@echo "TOOL_EXE_F:	$(TOOL_EXE_F)"
	@echo "=========================================="
	
	@echo -e $(GREEN)"Show all LIB-file message:"$(WHITE)
	@echo "LIB_F:	$(LIB_F)"
	@echo "=========================================="
	@echo -e $(GREEN)"-------------------- Make Show Success --------------------"

# ||=====================================================================||
# || 01. LIBRARY_PATH环境变量，指定程序静态链接库文件搜索路径；          ||
# ||     LD_LIBRARY_PATH环境变量，指定程序动态链接库文件搜索路径；       ||
# || 02. gcc (-I/ -L/ -Wl,-rpath=) *.o -o exe -lxxx;                     ||
# || 03. makefile一些常见的检查规则                                      ||
# ||         (1) --just-print, 不执行参数，只打印命令，不管目标是否跟新；||
# ||         (2) --what-if=<file>, 指定一个文件，一般和"-n"一起使用，来查||
# ||             这个文件所发生的规则命令；                              ||
# || 0.4 ANSI code sequence, control color                               ||
# ||=====================================================================||
