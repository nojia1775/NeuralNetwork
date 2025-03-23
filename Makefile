CXX = c++
CXXFLAGS = -Wall -Wextra -Werror -g -MMD

SRCS =	src/Input.cpp \
	src/HiddenCell.cpp \
	src/Output.cpp \
	src/network/NeuralNetwork.cpp \
	src/network/Json.cpp \
	src/network/BackPropagation.cpp \
	src/network/FeedForward.cpp

OBJS_DIR = obj/
OBJS = $(SRCS:%.cpp=$(OBJS_DIR)%.o)

NAME = neuralnetwork.a

DEPS = $(OBJS:.o=.d)

all: $(NAME)

$(NAME): $(OBJS)
	ar -rsc $(NAME) $(OBJS)

$(OBJS_DIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -fr $(OBJS_DIR)

fclean: clean
	rm -rf $(NAME)

re: fclean all

-include $(DEPS)

.PHONY: all re clean fclean
