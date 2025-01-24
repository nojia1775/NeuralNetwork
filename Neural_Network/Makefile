CXX = c++
CXXFLAGS = -Wall -Wextra -Werror -g

SRCS =	src/Input.cpp \
	src/HiddenCell.cpp \
	src/Output.cpp \
	src/NeuralNetwork.cpp

OBJS_DIR = obj/
OBJS = $(SRCS:src/%.cpp=$(OBJS_DIR)%.o)

NAME = neuralnetwork.a

all: $(NAME)

$(NAME): $(OBJS)
	ar -rsc $(NAME) $(OBJS)

$(OBJS_DIR)%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -fr $(OBJS_DIR)

fclean: clean
	rm -rf $(NAME)

re: fclean all

.PHONY: all re clean fclean
