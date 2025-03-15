import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const formVariants = cva("space-y-4", {
  variants: {
    layout: {
      vertical: "flex flex-col gap-4",
      horizontal: "flex flex-row items-center gap-4",
    },
    size: {
      sm: "text-sm",
      md: "text-base",
      lg: "text-lg",
    },
  },
  defaultVariants: {
    layout: "vertical",
    size: "md",
  },
});

export interface FormProps
  extends React.FormHTMLAttributes<HTMLFormElement>,
    VariantProps<typeof formVariants> {
  asChild?: boolean;
}

const Form = React.forwardRef<HTMLFormElement, FormProps>(
  ({ className, layout, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "form";
    return (
      <Comp
        ref={ref}
        className={cn(formVariants({ layout, size }), className)}
        {...props}
      />
    );
  }
);
Form.displayName = "Form";

export { Form };
