CXX = c++

CXXFLAGS = -Wall -Wextra -Werror

OBJS_DIR = obj

SRCS =	main.cpp

OBJS = $(SRCS:%.cpp=$(OBJS_DIR)/%.o)

NAME = prog

$(NAME): $(OBJS)
	@make -C Neural_Network
	$(CXX) $(CXXFLAGS) $^ -o $@ Neural_Network/neuralnetwork.a -g -lm
	
$(OBJS_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJS_DIR):
	mkdir -p $(OBJS_DIR)

all: $(NAME)
	@make -C Neural_Network

clean:
	@make clean -C Neural_Network
	rm -rf $(OBJS_DIR)

fclean: clean
	rm -rf $(NAME)
	@make fclean -C Neural_Network

re: fclean all

.PHONY: all clean fclean re